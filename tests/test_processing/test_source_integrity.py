from dataclasses import replace
from hashlib import md5, sha256
import struct

import pytest

from pbpk_kb.parsers.models import ParseAsset
from pbpk_kb.parsers.raw_formats import RawFormatParsers
from pbpk_kb.parsers.source_integrity import verified_unavailable_ole


def fixture_asset(tmp_path, *, directory_sector=534):
    contents = bytearray(163840)
    contents[:8] = bytes.fromhex("d0cf11e0a1b11ae1")
    struct.pack_into("<HHHH", contents, 24, 0x3E, 3, 0xFFFE, 9)
    struct.pack_into("<I", contents, 48, directory_sector)
    path = tmp_path / "supplement.xls"
    path.write_bytes(contents)
    return ParseAsset.from_inventory({
        "run_id": "run_test", "raw_asset_id": "asset:test", "source_id": "pmc_oa",
        "storage_path": str(path), "source_url": "https://pmc-oa-opendata.s3.amazonaws.com/PMC123.1/supplement.xls",
        "checksum": sha256(contents).hexdigest(), "file_size": len(contents),
        "file_extension": ".xls", "file_family": "table",
        "asset_metadata_json": {"byte_count": len(contents), "status_code": 200,
            "metadata": {"s3_bucket": "pmc-oa-opendata", "s3_key": "PMC123.1/supplement.xls",
                "s3_etag": md5(contents, usedforsecurity=False).hexdigest(), "s3_size": len(contents),
                "transfer_method": "official_pmc_static_object_plan"}},
    })


def test_proven_official_damage_is_unavailable_not_parsed(tmp_path):
    asset = fixture_asset(tmp_path)
    before = asset.physical_path.read_bytes()
    outcome = RawFormatParsers().parse_table(asset, None)
    assert outcome.status == "unsupported"
    assert outcome.metadata["source_content_unavailable"]
    assert outcome.metadata["minimum_required_bytes"] == 274432
    assert outcome.metadata["observation_extraction_blocked"]
    assert asset.physical_path.read_bytes() == before


@pytest.mark.parametrize("kind", ["size", "sha256", "etag", "multipart_etag", "http_error", "missing_metadata", "url", "key", "source", "archive", "header", "version", "length"])
def test_unproven_damage_is_not_silently_accepted(tmp_path, kind):
    asset = fixture_asset(tmp_path)
    if kind == "size":
        asset.asset_metadata_json["metadata"]["s3_size"] += 512
    elif kind == "sha256":
        asset = replace(asset, checksum="0" * 64)
    elif kind == "etag":
        asset.asset_metadata_json["metadata"]["s3_etag"] = "0" * 32
    elif kind == "multipart_etag":
        asset.asset_metadata_json["metadata"]["s3_etag"] += "-2"
    elif kind == "http_error":
        asset.asset_metadata_json["status_code"] = 503
    elif kind == "missing_metadata":
        asset.asset_metadata_json.pop("metadata")
    elif kind == "url":
        asset = replace(asset, source_url="https://other.example/PMC123.1/supplement.xls")
    elif kind == "key":
        asset.asset_metadata_json["metadata"]["s3_key"] = "PMC124.1/supplement.xls"
    elif kind == "source":
        asset = replace(asset, source_id="unknown")
    elif kind == "archive":
        asset = replace(asset, inner_path="supplement.xls")
    elif kind == "header":
        asset.physical_path.write_bytes(b"invalid" * 23406)
    elif kind == "version":
        contents = bytearray(asset.physical_path.read_bytes())
        struct.pack_into("<H", contents, 30, 13)
        asset.physical_path.write_bytes(contents)
    elif kind == "length":
        asset.physical_path.write_bytes(asset.physical_path.read_bytes()[:-512])
    assert verified_unavailable_ole(asset) is None


def test_in_bounds_directory_is_not_classified_corrupt(tmp_path):
    assert verified_unavailable_ole(fixture_asset(tmp_path, directory_sector=1)) is None
