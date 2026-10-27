from __future__ import annotations

from pathlib import Path
import zipfile


GENERIC_SUFFIXES = {".bin", ".dat", ".unk", ".fcgi"}
OOXML_CANONICAL_EXTENSIONS = {
    "document": ".docx",
    "presentation": ".pptx",
    "table": ".xlsx",
}


def extension_for(path: Path, content_type: str | None, url: str | None) -> str:
    name = path.name.lower()
    if name.endswith(".tsv.gz"):
        return ".tsv_gz"
    if name.endswith(".tar.gz") or name.endswith(".tar_gz"):
        return ".tar_gz"
    if path.suffix and path.suffix.lower() not in GENERIC_SUFFIXES:
        return path.suffix.lower()
    if url:
        url_suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if url_suffix and url_suffix not in GENERIC_SUFFIXES:
            return url_suffix
    lowered = (content_type or "").lower()
    if "json" in lowered:
        return ".json"
    if "xml" in lowered:
        return ".xml"
    if "html" in lowered:
        return ".html"
    if "pdf" in lowered:
        return ".pdf"
    return path.suffix.lower()


def file_family_for(
    path: Path,
    extension: str,
    content_type: str | None,
    source_id: str | None,
) -> str:
    if path.is_dir():
        return "repo"
    lowered_ct = (content_type or "").lower()
    name = path.name.lower()

    if extension == ".json" or "json" in lowered_ct:
        return "json"
    if extension == ".xml" or "xml" in lowered_ct:
        return "xml"
    if extension in {".html", ".htm", ".aspx"} or "html" in lowered_ct:
        return "html"
    if extension == ".pdf" or "pdf" in lowered_ct:
        return "pdf"
    if extension in {".csv", ".tsv", ".tsv_gz", ".xlsx", ".xls", ".xpt"} or name.endswith(".tsv.gz"):
        return "table"
    if extension in {".doc", ".docx", ".rtf", ".odt"}:
        return "document"
    if extension in {".ppt", ".pptx", ".odp"}:
        return "presentation"
    if extension in {".zip", ".gz", ".tgz", ".tar", ".tar_gz", ".rar", ".7z"}:
        return "archive"
    if extension in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".eps", ".webp"} or lowered_ct.startswith("image/"):
        return "image"
    if extension in {".mp3", ".mp4", ".avi", ".mov", ".wav"} or lowered_ct.startswith(("audio/", "video/")):
        return "media"
    if extension in {".sqlite", ".sqlite3", ".db"} or "sqlite" in lowered_ct:
        return "database"
    if extension == ".pzf":
        return "specialized_binary"
    if extension in {".txt", ".css", ".md"} or "text/plain" in lowered_ct:
        return "text"
    if source_id == "osp_pbpk_model_library" and extension in {".pksim5", ".pkml", ".mobi"}:
        return "repo"
    return sniff_file_family(path)


def sniff_file_family(path: Path) -> str:
    if not path.is_file():
        return "unknown"
    try:
        with path.open("rb") as file:
            sample = file.read(8192)
    except OSError:
        return "unknown"
    return sniff_bytes_family(sample)


def sniff_bytes_family(sample: bytes) -> str:
    stripped = sample.lstrip()
    lowered = stripped[:512].lower()
    if sample.startswith(b"%PDF"):
        return "pdf"
    if sample.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08", b"\x1f\x8b")):
        return "archive"
    if sample.startswith((b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00", b"7z\xbc\xaf\x27\x1c")):
        return "archive"
    if sample.startswith(b"SQLite format 3\x00"):
        return "database"
    if sample.startswith(b"PCFFGRA4"):
        return "specialized_binary"
    if sample.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"II*\x00", b"MM\x00*")):
        return "image"
    if len(sample) >= 12 and sample.startswith(b"RIFF") and sample[8:12] == b"WEBP":
        return "image"
    if sample.startswith(b"ID3") or (len(sample) >= 12 and sample[4:8] == b"ftyp"):
        return "media"
    if len(sample) >= 12 and sample.startswith(b"RIFF") and sample[8:12] in {b"AVI ", b"WAVE"}:
        return "media"
    if sample.startswith(bytes.fromhex("d0cf11e0a1b11ae1")):
        return "table"
    if stripped.startswith((b"{", b"[")):
        return "json"
    if lowered.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        return "html"
    if stripped.startswith(b"<"):
        return "xml"
    if sample and b"\x00" not in sample:
        decoded = sample.decode("utf-8", errors="replace")
        replacement_ratio = decoded.count("\ufffd") / max(1, len(decoded))
        if replacement_ratio < 0.02:
            return "text"
    return "unknown"


def ooxml_package_family(path: Path) -> str | None:
    if not path.is_file() or not zipfile.is_zipfile(path):
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            content_types = archive.read("[Content_Types].xml").lower()
    except (KeyError, OSError, zipfile.BadZipFile):
        return None
    if b"spreadsheetml.sheet.main+xml" in content_types or b"ms-excel.sheet.macroenabled.main+xml" in content_types:
        return "table"
    if b"wordprocessingml.document.main+xml" in content_types or b"ms-word.document.macroenabled.main+xml" in content_types:
        return "document"
    if b"presentationml.presentation.main+xml" in content_types or b"ms-powerpoint.presentation.macroenabled.main+xml" in content_types:
        return "presentation"
    return None
