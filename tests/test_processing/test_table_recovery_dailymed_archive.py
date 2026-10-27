"""Exact official ZIP source recovery, without extracting or changing raw bytes."""
from __future__ import annotations

import json
import warnings
from hashlib import sha256
from pathlib import Path
from zipfile import BadZipFile, ZIP_STORED, ZipFile

import pytest

from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
from pbpk_kb.processing.table_recovery_stage import (
    DAILYMED_ARCHIVE_READER_VERSION, TableRecoveryStage, read_dailymed_xml_source,
)

SETID = "8ae4a0c1-1424-47a9-9a59-7fe38bedc0c7"
MEMBER = SETID + ".xml"
XML = f'''<document xmlns="urn:hl7-org:v3">
<id root="{SETID}"/><setId root="{SETID}"/><versionNumber value="1"/>
<component><structuredBody><component><section><title>Pharmacokinetics</title><text>
<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>
<tbody><tr><td>Cmax</td><td>12 ng/mL</td></tr></tbody></table>
</text></section></component></structuredBody></component>
<subject><manufacturedProduct><manufacturedProduct><ingredient classCode="ACTIM">
<ingredientSubstance><name>Gabapentin</name></ingredientSubstance>
</ingredient></manufacturedProduct></manufacturedProduct></subject></document>'''.encode()


def fixture(tmp_path: Path, xml: bytes = XML):
    path = tmp_path / "label.zip"
    with ZipFile(path, "w", compression=ZIP_STORED) as z:
        z.writestr(MEMBER, xml)
    asset = {
        "run_id": "run_test", "raw_asset_id": "archive", "source_id": "dailymed", "compound": "gabapentin",
        "storage_path": str(path), "file_family": "archive", "file_extension": ".zip",
        "source_url": f"https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?type=zip&setid={SETID}&version=1",
        "checksum": sha256(path.read_bytes()).hexdigest(), "file_size": path.stat().st_size,
        "asset_metadata_json": {"metadata": {
            "recovery_method": "verified_official_exact_version_archive", "setid": SETID, "spl_version": 1,
            "member_path": MEMBER, "member_sha256": sha256(xml).hexdigest(), "member_bytes": len(xml),
            "document_id": SETID, "archive_member_count": 1, "archive_expanded_bytes": len(xml),
        }},
    }
    chunk = {key: asset[key] for key in ("run_id", "raw_asset_id", "source_id", "compound", "storage_path", "source_url", "checksum")}
    chunk.update(chunk_id="chunk", parsed_document_id="document", processing_status="deferred_unparsed_flattened_table",
                 source_location=f"dailymed:label.zip!{MEMBER}#xml_path=//section[1]#chunk=1")
    return asset, chunk


def write_rows(layout, name, rows):
    layout.path(name, for_write=True).write_text("".join(json.dumps(r) + "\n" for r in rows))


def rows(layout, name):
    return [json.loads(line) for line in layout.path(name).read_text().splitlines()]


def stage_fixture(tmp_path, asset, chunk, existing=False):
    layout = raw_processing_layout(tmp_path / "processed")
    layout.ensure_dirs()
    write_rows(layout, "raw_inventory.jsonl", [asset])
    write_rows(layout, "chunk_processing_status.jsonl", [chunk])
    tables = []
    if existing:
        proof = asset["asset_metadata_json"]["metadata"]
        tables = [{**{key: chunk[key] for key in ("run_id", "raw_asset_id", "source_id", "compound", "storage_path", "source_url", "checksum")},
                   "parsed_table_id": "existing", "source_location": chunk["source_location"].replace("#chunk=1", "#table=1"),
                   "columns_json": ["Parameter", "Value"], "cells_json": [["Cmax", "12 ng/mL"]],
                   "metadata_json": {"archive_inner_path": MEMBER, "archive_member_checksum": proof["member_sha256"]}}]
    write_rows(layout, "parsed_tables.jsonl", tables)
    write_rows(layout, "evidence_candidates_deterministic.jsonl", [])
    write_rows(layout, "evidence_candidates_llm.jsonl", [])
    layout.path("pipeline_state.json", for_write=True).write_text(json.dumps({"run_id": "run_test", "stages": {
        "llm_extract": {"status": "completed", "accepted": True},
        "completeness": {"status": "not_started"}, "embedding": {"status": "not_started"},
    }}))
    return TableRecoveryStage(output_dir=layout.root), layout


@pytest.mark.parametrize("existing", [False, True])
def test_archive_recovers_tables_and_active_ingredient_with_exact_provenance(tmp_path, existing):
    asset, chunk = fixture(tmp_path)
    before = Path(asset["storage_path"]).read_bytes()
    stage, layout = stage_fixture(tmp_path, asset, chunk, existing)
    summary = stage.run()
    assert summary["accepted"] is True
    assert summary["recovered_table_count"] == 1 and summary["table_evidence_count"] == 1
    assert rows(layout, "table_recovery_status.jsonl")[0]["terminal_status"] == (
        "resolved_by_existing_table" if existing else "recovered_from_raw")
    for name in ("recovered_tables.jsonl", "recovered_table_rows.jsonl", "table_evidence_candidates.jsonl"):
        record, = rows(layout, name)
        assert record["source_location"].startswith(f"dailymed:label.zip!{MEMBER}#")
        assert record["checksum"] == asset["checksum"]
        assert record["metadata_json"]["archive_inner_path"] == MEMBER
        assert record["metadata_json"]["archive_member_checksum"] == sha256(XML).hexdigest()
        assert record["metadata_json"]["archive_source_reader_version"] == DAILYMED_ARCHIVE_READER_VERSION
    candidate, = rows(layout, "table_evidence_candidates.jsonl")
    assert (candidate["compound"], candidate["parameter"], candidate["value"], candidate["unit"]) == (
        "gabapentin", "Cmax", 12.0, "ng/mL")
    assert candidate["curation_status"] == "needs_review"
    assert Path(asset["storage_path"]).read_bytes() == before
    assert not (tmp_path / MEMBER).exists()
    # A previously cached successful ownership must not bypass member identity.
    assert stage._resolve_asset_ownership(asset, {"archive_inner_path": "wrong.xml"})[0] is None


@pytest.mark.parametrize("field,value", [
    ("recovery_method", "unregistered"), ("member_path", "other.xml"), ("member_sha256", "0" * 64),
    ("member_bytes", 1), ("document_id", "00000000-0000-0000-0000-000000000000"),
    ("setid", "00000000-0000-0000-0000-000000000000"), ("spl_version", 2),
    ("archive_member_count", 2), ("archive_expanded_bytes", 1),
])
def test_registered_archive_identity_changes_remain_failures(tmp_path, field, value):
    asset, chunk = fixture(tmp_path)
    asset["asset_metadata_json"]["metadata"][field] = value
    stage, layout = stage_fixture(tmp_path, asset, chunk)
    summary = stage.run()
    assert summary["accepted"] is False
    assert summary["terminal_status_counts"] == {"recovery_failed": 1}
    assert rows(layout, "table_evidence_candidates.jsonl") == []
    assert rows(layout, "recovered_tables.jsonl") == []


@pytest.mark.parametrize("field,value", [
    ("checksum", "0" * 64), ("file_size", 1), ("source_url", "https://example.test/label.zip"),
])
def test_archive_bytes_and_official_url_are_revalidated(tmp_path, field, value):
    asset, _ = fixture(tmp_path)
    asset[field] = value
    with pytest.raises(ValueError):
        read_dailymed_xml_source(asset)


@pytest.mark.parametrize("field,value", [
    ("source_location", "dailymed:label.zip!another.xml#xml_path=//section[1]#chunk=1"),
    ("source_location", "dailymed:label.zip#xml_path=//section[1]#chunk=1"),
    ("checksum", "0" * 64), ("source_url", "https://example.test/not-source"),
])
def test_chunk_reference_mismatch_cannot_be_covered(tmp_path, field, value):
    asset, chunk = fixture(tmp_path)
    chunk[field] = value
    stage, layout = stage_fixture(tmp_path, asset, chunk)
    summary = stage.run()
    assert summary["accepted"] is False
    assert rows(layout, "recovered_tables.jsonl") == []


def test_existing_table_member_provenance_mismatch_aborts_before_emitting(tmp_path):
    asset, chunk = fixture(tmp_path)
    stage, layout = stage_fixture(tmp_path, asset, chunk, True)
    record, = rows(layout, "parsed_tables.jsonl")
    record["metadata_json"]["archive_member_checksum"] = "0" * 64
    write_rows(layout, "parsed_tables.jsonl", [record])
    with pytest.raises(ValueError, match="provenance"):
        stage.run()
    assert not layout.path("recovered_tables.jsonl").exists()


def test_reusing_valid_table_cannot_cover_a_different_member_chunk(tmp_path):
    asset, chunk = fixture(tmp_path)
    stage, layout = stage_fixture(tmp_path, asset, chunk, True)
    chunk["source_location"] = chunk["source_location"].replace(MEMBER, "other.xml")
    write_rows(layout, "chunk_processing_status.jsonl", [chunk])
    with pytest.raises(ValueError, match="registered source/member"):
        stage.run()
    assert not layout.path("recovered_tables.jsonl").exists()


@pytest.mark.parametrize("unsafe_member", ["../other.xml", "/absolute.xml", MEMBER, "second.xml"])
def test_unsafe_duplicate_or_multiple_xml_archives_are_rejected(tmp_path, unsafe_member):
    asset, _ = fixture(tmp_path)
    path = Path(asset["storage_path"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(path, "a") as z:
            z.writestr(unsafe_member, XML)
    asset["checksum"] = sha256(path.read_bytes()).hexdigest()
    asset["file_size"] = path.stat().st_size
    with pytest.raises(ValueError):
        read_dailymed_xml_source(asset)


def test_member_crc_is_checked_even_when_outer_digest_is_registered(tmp_path):
    asset, _ = fixture(tmp_path)
    path = Path(asset["storage_path"])
    data = path.read_bytes()
    assert b"12 ng/mL" in data
    path.write_bytes(data.replace(b"12 ng/mL", b"13 ng/mL", 1))
    asset["checksum"] = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(BadZipFile, match="CRC"):
        read_dailymed_xml_source(asset)


@pytest.mark.parametrize("xml", [
    b'<!DOCTYPE document [<!ENTITY x "bad">]>' + XML,
    XML.replace(b'value="1"', b'value="2"'),
    XML.replace(b"<document ", b"<not-document ").replace(b"</document>", b"</not-document>"),
])
def test_dtd_and_incorrect_spl_identity_remain_rejected(tmp_path, xml):
    asset, _ = fixture(tmp_path, xml)
    with pytest.raises(ValueError):
        read_dailymed_xml_source(asset)
