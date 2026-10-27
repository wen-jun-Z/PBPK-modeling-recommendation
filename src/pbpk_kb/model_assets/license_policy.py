from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from pbpk_kb.model_assets.models import LicenseDecision, LicenseStatus


LICENSE_FILENAMES = (
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "COPYING",
    "COPYING.md",
    "COPYING.txt",
    "Open Systems Pharmacology Suite License.pdf",
)

SPDX_MARKERS = {
    "apache license": "Apache-2.0",
    "mit license": "MIT",
    "gnu general public license version 2": "GPL-2.0-only",
    "gnu general public license version 3": "GPL-3.0-only",
    "creative commons attribution 4.0": "CC-BY-4.0",
    "cc by 4.0": "CC-BY-4.0",
    "gplv2 license": "GPL-2.0-only",
    "gpl v2 license": "GPL-2.0-only",
    "gplv3 license": "GPL-3.0-only",
    "gpl v3 license": "GPL-3.0-only",
}


def _normalized_license_text(path: Path) -> str:
    if path.suffix.casefold() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", text).casefold(),
    ).strip()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repository_license(repository_root: Path) -> LicenseDecision:
    for filename in LICENSE_FILENAMES:
        path = repository_root / filename
        if not path.is_file():
            continue
        try:
            text = _normalized_license_text(path)
        except Exception:
            return LicenseDecision(
                status=LicenseStatus.UNRESOLVED,
                source_location=filename,
                source_checksum=sha256_file(path),
                reason_code="LICENSE_TEXT_UNREADABLE",
            )
        spdx_id = next((spdx for marker, spdx in SPDX_MARKERS.items() if marker in text), None)
        if spdx_id is None:
            return LicenseDecision(
                status=LicenseStatus.UNRESOLVED,
                source_location=filename,
                source_checksum=sha256_file(path),
                reason_code="LICENSE_TEXT_UNRECOGNIZED",
            )
        return LicenseDecision(
            status=LicenseStatus.ALLOWED,
            spdx_id=spdx_id,
            source_location=filename,
            source_checksum=sha256_file(path),
            reason_code="REPOSITORY_LICENSE_EXPLICIT",
        )
    for filename in ("README.md", "README.txt", "README.rst", "README"):
        path = repository_root / filename
        if not path.is_file():
            continue
        try:
            text = _normalized_license_text(path)
        except Exception:
            continue
        explicit_model_statement = (
            "model" in text
            and ("distributed under" in text or "licensed under" in text)
        )
        spdx_id = next(
            (spdx for marker, spdx in SPDX_MARKERS.items() if marker in text),
            None,
        )
        if explicit_model_statement and spdx_id is not None:
            return LicenseDecision(
                status=LicenseStatus.ALLOWED,
                spdx_id=spdx_id,
                source_location=filename,
                source_checksum=sha256_file(path),
                reason_code="MODEL_LICENSE_STATEMENT_EXPLICIT",
            )
    return LicenseDecision(
        status=LicenseStatus.UNRESOLVED,
        reason_code="LICENSE_FILE_MISSING",
    )
