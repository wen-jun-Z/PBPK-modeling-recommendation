from __future__ import annotations

import binascii
import struct
import zlib
from dataclasses import dataclass
from typing import Literal


LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
ZIP64_EXTRA = struct.Struct("<HHQQ")
ZIP_LOCAL_SIGNATURE = 0x04034B50
DEFLATE_METHOD = 8
MAX_CONTENT_BYTES = 256 * 1024 * 1024


class OSPContentError(ValueError):
    pass


@dataclass(frozen=True)
class DecompressedOSPContent:
    filename: str
    data: bytes
    storage_format: Literal["zip64_deflate", "zip_deflate", "utf8_xml"]
    compressed_size: int
    uncompressed_size: int
    crc32: str
    crc_valid: bool | None


def decompress_osp_content(blob: bytes) -> DecompressedOSPContent:
    """Decode a compressed or inline XML payload stored in OSP CONTENTS.Data."""

    if blob.startswith((b"<?xml", b"\xef\xbb\xbf<?xml")):
        actual_crc = binascii.crc32(blob) & 0xFFFFFFFF
        return DecompressedOSPContent(
            filename="inline.xml",
            data=blob,
            storage_format="utf8_xml",
            compressed_size=len(blob),
            uncompressed_size=len(blob),
            crc32=f"{actual_crc:08x}",
            crc_valid=None,
        )

    if len(blob) < LOCAL_FILE_HEADER.size:
        raise OSPContentError(
            "OSP content is neither inline UTF-8 XML nor a ZIP local stream."
        )
    (
        signature,
        _version,
        flags,
        method,
        _mtime,
        _mdate,
        expected_crc,
        compressed_size_32,
        uncompressed_size_32,
        filename_length,
        extra_length,
    ) = LOCAL_FILE_HEADER.unpack_from(blob)
    if signature != ZIP_LOCAL_SIGNATURE:
        raise OSPContentError(f"Unexpected content signature: {signature:#x}.")
    if method != DEFLATE_METHOD:
        raise OSPContentError(f"Unsupported OSP compression method: {method}.")
    if flags & 0x09:
        raise OSPContentError("Encrypted or descriptor-based OSP content is unsupported.")

    filename_start = LOCAL_FILE_HEADER.size
    filename_end = filename_start + filename_length
    extra_end = filename_end + extra_length
    if extra_end > len(blob):
        raise OSPContentError("OSP content header extends beyond the stored blob.")
    filename = blob[filename_start:filename_end].decode("utf-8", "strict")
    extra = blob[filename_end:extra_end]
    if 0xFFFFFFFF in (compressed_size_32, uncompressed_size_32):
        if len(extra) < ZIP64_EXTRA.size:
            raise OSPContentError("OSP content does not contain the expected ZIP64 sizes.")
        extra_id, payload_size, uncompressed_size, compressed_size = ZIP64_EXTRA.unpack_from(extra)
        if extra_id != 0x0001 or payload_size != 16:
            raise OSPContentError("OSP content ZIP64 extra field is invalid.")
        storage_format = "zip64_deflate"
    else:
        compressed_size, uncompressed_size = compressed_size_32, uncompressed_size_32
        storage_format = "zip_deflate"
    if uncompressed_size > MAX_CONTENT_BYTES:
        raise OSPContentError("OSP content exceeds the decompression size limit.")
    compressed = blob[extra_end:]
    if compressed_size != len(compressed):
        raise OSPContentError(
            f"Compressed size mismatch: header={compressed_size}, stored={len(compressed)}."
        )
    try:
        decoder = zlib.decompressobj(-zlib.MAX_WBITS)
        data = decoder.decompress(compressed, min(MAX_CONTENT_BYTES, uncompressed_size) + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise OSPContentError("OSP content is truncated, oversized, or contains trailing data.")
    except zlib.error as exc:
        raise OSPContentError(f"OSP deflate stream is invalid: {exc}") from exc
    if len(data) != uncompressed_size:
        raise OSPContentError(
            f"Uncompressed size mismatch: header={uncompressed_size}, decoded={len(data)}."
        )
    actual_crc = binascii.crc32(data) & 0xFFFFFFFF
    return DecompressedOSPContent(
        filename=filename,
        data=data,
        storage_format=storage_format,
        compressed_size=compressed_size,
        uncompressed_size=uncompressed_size,
        crc32=f"{actual_crc:08x}",
        crc_valid=actual_crc == expected_crc,
    )
