from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest
from defusedxml.common import DefusedXmlException
from xml.etree.ElementTree import ParseError
from xml.sax import SAXParseException

from pbpk_kb.parsers.emitter import ParseEmitter
from pbpk_kb.parsers.models import ParseAsset
from pbpk_kb.parsers.raw_formats import (
    PUBLISHER_AVAILABILITY_MAX_BYTES,
    PUBLISHER_AVAILABILITY_METADATA,
    RawFormatParsers,
    recognize_publisher_data_availability,
)
from pbpk_kb.processing.deterministic_extract_stage import DeterministicExtractStage
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout


METADATA_XML = (
    b"<data:data><datasets><author>author@example.test</author><type>0</type>"
    b"<dataset><value>1</value><reason>Data will be made available on request</reason>"
    b"<comments/></dataset></datasets></data:data>"
)
EXPECTED_RECORD = {
    "availability_statement": "Data will be made available on request",
    "publisher_type": "0",
    "publisher_value": "1",
}


def _asset(tmp_path: Path, payload: bytes, source_id: str = "pmc_oa") -> ParseAsset:
    path = tmp_path / "mmc1.xml"
    path.write_bytes(payload)
    return ParseAsset.from_inventory({
        "run_id": "run_metadata_test", "raw_asset_id": "asset:publisher-metadata",
        "source_id": source_id, "storage_path": str(path), "file_family": "xml",
        "file_extension": ".xml", "file_size": len(payload), "checksum": sha256(payload).hexdigest(),
        "source_url": "https://example.test/PMC123/mmc1.xml", "asset_metadata_json": {},
    })


def _rows(layout, name: str) -> list[dict]:
    return [json.loads(line) for line in layout.path(name).read_text().splitlines()]


@pytest.mark.parametrize("payload", [
    METADATA_XML,
    b'<?xml version="1.0" encoding="UTF-8"?>\n' + METADATA_XML.replace(b"><", b">\n<"),
])
def test_metadata_is_provenanced_but_never_scientific_evidence(tmp_path: Path, payload: bytes) -> None:
    asset = _asset(tmp_path, payload)
    layout = raw_processing_layout(tmp_path / "processed")
    assert recognize_publisher_data_availability(payload) == EXPECTED_RECORD
    with ParseEmitter(layout, asset.run_id) as emitter:
        outcome = RawFormatParsers().parse_pmc_xml(asset, emitter)
        assert dict(emitter.counts) == {"parsed_records": 1}
    assert outcome.status == "metadata_only"
    assert outcome.metadata == PUBLISHER_AVAILABILITY_METADATA
    assert asset.physical_path.read_bytes() == payload
    record, = _rows(layout, "parsed_records.jsonl")
    assert record["record_type"] == "publisher_data_availability"
    assert record["parser_name"] == "pmc_jats_parser"
    assert record["record_json"] == EXPECTED_RECORD
    assert record["metadata_json"] == PUBLISHER_AVAILABILITY_METADATA
    for name in ("run_id", "raw_asset_id", "source_id", "storage_path", "source_url", "checksum"):
        assert record[name] == getattr(asset, name)
    assert record["source_location"].endswith("#xml_path=/data:data/datasets/dataset")
    for name in ParseEmitter.OUTPUT_NAMES:
        if name != "parsed_records.jsonl":
            assert _rows(layout, name) == []
    mapping = DeterministicExtractStage(run_root=tmp_path, output_dir=tmp_path / "processed")._map_record(record)
    assert mapping.status == "not_evidence"
    assert mapping.evidence == []
    assert mapping.pointers == []


@pytest.mark.parametrize("payload", [
    METADATA_XML.replace(b"data:data", b"other:data"),
    METADATA_XML.replace(b"<value>1</value>", b"<x:value>1</x:value>"),
    METADATA_XML.replace(b"<type>0</type>", b"<type>1</type>"),
    METADATA_XML.replace(b"<value>1</value>", b"<value>2</value>"),
    METADATA_XML.replace(b"<value>1</value>", b"<value>1</value><value>1</value>"),
    METADATA_XML.replace(b"<value>1</value>", b"<value dose='1'>1</value>"),
    METADATA_XML.replace(b"<type>0</type>", b""),
    METADATA_XML.replace(b"<comments/>", b"<comments>Cmax = 10 ng/ml</comments>"),
    METADATA_XML.replace(b"<comments/>", b"<body><p>Cmax = 10 ng/ml</p></body><comments/>"),
    METADATA_XML.replace(b"<comments/>", b"<comments><table>1</table></comments>"),
    METADATA_XML.replace(b"Data will be made available on request", b"Cmax = 10 ng/ml"),
    METADATA_XML.replace(b"<dataset>", b"<dataset>Cmax = 10 ng/ml"),
    METADATA_XML.replace(b"<comments/>", b"<comments/>Cmax = 10 ng/ml"),
    METADATA_XML.replace(b"<comments/>", b"<!-- Cmax = 10 ng/ml --><comments/>"),
    METADATA_XML.replace(b"<comments/>", b"<?application extract?><comments/>"),
    METADATA_XML.replace(b"<author>author@example.test</author>", b"<author/>"),
    METADATA_XML.replace(b"<data:data>", b"<data:data id='parameter'>"),
    METADATA_XML.replace(b"<datasets>", b"<datasets>" + b" " * PUBLISHER_AVAILABILITY_MAX_BYTES),
    METADATA_XML.replace(b"<comments/>", b"<!-- hidden content --><comments/>").decode().encode("utf-16"),
])
def test_unrecognized_shapes_remain_failures_without_partial_outputs(tmp_path: Path, payload: bytes) -> None:
    assert recognize_publisher_data_availability(payload) is None
    asset = _asset(tmp_path, payload)
    layout = raw_processing_layout(tmp_path / "processed")
    with ParseEmitter(layout, asset.run_id) as emitter:
        with pytest.raises(ParseError, match="unbound prefix"):
            RawFormatParsers().parse_pmc_xml(asset, emitter)
        assert not emitter.counts
    assert asset.physical_path.read_bytes() == payload


@pytest.mark.parametrize("prefix", [
    b'<!DOCTYPE data:data>',
    b'<!DOCTYPE data:data [<!ENTITY e "injected">]>',
    b'<!DOCTYPE data:data [<!ENTITY e SYSTEM "file:///must-not-read">]>',
    b'<!DOCTYPE data:data SYSTEM "https://must-not-fetch.invalid/source.dtd">',
])
def test_dtd_and_entity_inputs_stay_forbidden(tmp_path: Path, prefix: bytes) -> None:
    payload = prefix + METADATA_XML
    with pytest.raises(DefusedXmlException):
        recognize_publisher_data_availability(payload)
    asset = _asset(tmp_path, payload)
    with ParseEmitter(raw_processing_layout(tmp_path / "processed"), asset.run_id) as emitter:
        with pytest.raises(DefusedXmlException):
            RawFormatParsers().parse_pmc_xml(asset, emitter)
        assert not emitter.counts


@pytest.mark.parametrize("payload", [METADATA_XML[:-12], METADATA_XML + b"<extra/>", METADATA_XML.replace(b"<comments/>", b"&undefined;")])
def test_incomplete_or_multiple_root_xml_is_not_recovered(tmp_path: Path, payload: bytes) -> None:
    asset = _asset(tmp_path, payload)
    with ParseEmitter(raw_processing_layout(tmp_path / "processed"), asset.run_id) as emitter:
        with pytest.raises((ParseError, SAXParseException)):
            RawFormatParsers().parse_pmc_xml(asset, emitter)
        assert not emitter.counts


def test_metadata_compatibility_is_only_available_for_pmc_source(tmp_path: Path) -> None:
    asset = _asset(tmp_path, METADATA_XML, "unregistered_source")
    with ParseEmitter(raw_processing_layout(tmp_path / "processed"), asset.run_id) as emitter:
        with pytest.raises(ParseError, match="unbound prefix"):
            RawFormatParsers().parse_pmc_xml(asset, emitter)
        assert not emitter.counts


def test_normal_jats_keeps_original_body_and_table_parser(tmp_path: Path, monkeypatch) -> None:
    payload = (
        b'<article><front><article-meta><article-title>PK study</article-title></article-meta></front>'
        b'<body><sec><title>Results</title><p>Observed exposure was measured in plasma.</p>'
        b'<table-wrap><table><tr><th>time</th><th>concentration</th></tr>'
        b'<tr><td>1</td><td>2</td></tr></table></table-wrap></sec></body></article>'
    )
    asset = _asset(tmp_path, payload)
    def forbidden_probe(_payload):
        raise AssertionError("valid JATS must not enter publisher metadata compatibility")
    monkeypatch.setattr("pbpk_kb.parsers.raw_formats.recognize_publisher_data_availability", forbidden_probe)
    layout = raw_processing_layout(tmp_path / "processed")
    with ParseEmitter(layout, asset.run_id) as emitter:
        outcome = RawFormatParsers().parse_pmc_xml(asset, emitter)
    assert outcome.status == "parsed"
    assert len(_rows(layout, "parsed_documents.jsonl")) == 1
    assert len(_rows(layout, "parsed_tables.jsonl")) == 1
    assert _rows(layout, "text_chunks.jsonl")
