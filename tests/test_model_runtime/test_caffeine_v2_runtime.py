from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from pbpk_kb.model_runtime.caffeine_v2_runtime import (
    EXPECTED_REACTIONS,
    _verify_windows_file_manifest,
)


def test_expected_reactions_preserve_three_named_products() -> None:
    assert EXPECTED_REACTIONS == {
        "Caffeine-CYP1A2-N3-Paraxanthine": (
            "Caffeine-CYP1A2-N3-Paraxanthine Metabolite"
        ),
        "Caffeine-CYP1A2-N1-Theobromine": (
            "Caffeine-CYP1A2-N1-Theobromine Metabolite"
        ),
        "Caffeine-CYP1A2-N7-Theophylline": (
            "Caffeine-CYP1A2-N7-Theophylline Metabolite"
        ),
    }


def test_windows_file_manifest_rejects_unsafe_path(tmp_path: Path) -> None:
    artifact = tmp_path / "result.csv"
    artifact.write_text("x\n1\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    rows = [
        {
            "RelativePath": "result.csv",
            "ByteSize": str(artifact.stat().st_size),
            "SHA256": digest,
        }
    ]
    with (tmp_path / "files.sha256.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checks, row_count = _verify_windows_file_manifest(tmp_path)

    assert row_count == 1
    assert checks["file_manifest_files_complete"]
    assert checks["file_manifest_sha256_match"]
    assert not checks["file_manifest_has_50_rows"]

    rows[0]["RelativePath"] = "../../result.csv"
    with (tmp_path / "files.sha256.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checks, _ = _verify_windows_file_manifest(tmp_path)

    assert not checks["file_manifest_paths_safe"]
