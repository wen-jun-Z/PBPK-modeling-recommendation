"""Recognize a narrowly proven, unavailable official source without inventing data.

An unsupported outcome is permitted only when the original official PMC object
was downloaded byte-for-byte and its OLE directory lies beyond the object. Local
truncation, unknown provenance and ordinary parser errors still fail normally.
"""
from __future__ import annotations

from hashlib import md5, sha256
import re
import struct
from urllib.parse import unquote, urlsplit

from pbpk_kb.parsers.models import ParseAsset, ParseOutcome


def verified_unavailable_ole(asset: ParseAsset) -> ParseOutcome | None:
    if asset.source_id != "pmc_oa" or asset.inner_path or asset.file_extension.lower() != ".xls":
        return None
    outer = asset.asset_metadata_json
    metadata = outer.get("metadata") or {}
    url = urlsplit(asset.source_url or "")
    key = metadata.get("s3_key")
    etag = str(metadata.get("s3_etag") or "").strip('"')
    if (url.scheme != "https" or url.netloc != "pmc-oa-opendata.s3.amazonaws.com"
            or url.query or url.fragment or not key or unquote(url.path) != "/" + key
            or metadata.get("s3_bucket") != "pmc-oa-opendata"
            or metadata.get("transfer_method") != "official_pmc_static_object_plan"
            or not re.fullmatch(r"[0-9a-f]{32}", etag)
            or outer.get("status_code") != 200):
        return None
    path = asset.physical_path
    size = path.stat().st_size
    if any(type(value) is not int or value != size for value in (
            asset.file_size, outer.get("byte_count"), metadata.get("s3_size"))):
        return None
    with path.open("rb") as stream:
        header = stream.read(512)
        if (len(header) != 512 or header[:8] != bytes.fromhex("d0cf11e0a1b11ae1")
                or header[28:30] != b"\xfe\xff"):
            return None
        major, sector_shift = struct.unpack_from("<H", header, 26)[0], struct.unpack_from("<H", header, 30)[0]
        if (major, sector_shift) not in {(3, 9), (4, 12)}:
            return None
        sector_bytes = 1 << sector_shift
        directory_sector = struct.unpack_from("<I", header, 48)[0]
        if directory_sector >= 0xFFFFFFFA:
            return None
        minimum_required_bytes = (directory_sector + 2) * sector_bytes
        if minimum_required_bytes <= size:
            return None
        # Do not weaken integrity checking to rescue a locally incomplete copy.
        digests = sha256(), md5(usedforsecurity=False)
        stream.seek(0)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            for digest in digests:
                digest.update(block)
    if digests[0].hexdigest() != asset.checksum or digests[1].hexdigest() != etag:
        return None
    return ParseOutcome(
        status="unsupported",
        warnings=["Official source has an out-of-bounds OLE directory; attachment content is unavailable. "
                  "Original bytes retained; no table or evidence extracted; a complete official replacement is required."],
        metadata={
            "source_content_unavailable": True,
            "reason": "official_ole_directory_out_of_bounds",
            "checksum_verified": True,
            "official_etag_verified": True,
            "observation_extraction_blocked": True,
            "source_url": asset.source_url,
            "sha256": asset.checksum,
            "official_etag": etag,
            "official_object_bytes": size,
            "ole_directory_sector": directory_sector,
            "ole_sector_bytes": sector_bytes,
            "minimum_required_bytes": minimum_required_bytes,
        },
    )
