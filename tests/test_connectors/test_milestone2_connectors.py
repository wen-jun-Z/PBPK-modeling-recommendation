from __future__ import annotations

import json
from pathlib import Path

from pbpk_kb.connectors.registry import CONNECTOR_REGISTRY
from pbpk_kb.connectors.run_connectors import run_sources
from pbpk_kb.connectors.dailymed import DailyMedConnector
from pbpk_kb.connectors.hpa import HumanProteinAtlasConnector


MILESTONE2_SOURCES = {
    "pubmed",
    "dailymed",
    "pubchem",
    "osp_pbpk_model_library",
    "pkdb",
}

STAGE_A_SOURCES = {
    "drugs_fda",
    "rxnorm",
    "drugcentral",
    "chembl",
    "clinicaltrials",
}

STAGE_A_PLUS_MILESTONE2_SOURCES = MILESTONE2_SOURCES | STAGE_A_SOURCES
EXPANDED_STAGE1_SOURCES = STAGE_A_PLUS_MILESTONE2_SOURCES | {"hpa"}


def _json_lines(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as records_file:
        return [json.loads(line) for line in records_file if line.strip()]


def test_milestone2_connector_registry_contains_required_sources() -> None:
    assert MILESTONE2_SOURCES.issubset(CONNECTOR_REGISTRY)


def test_stage_a_connector_registry_contains_10_sources() -> None:
    assert STAGE_A_PLUS_MILESTONE2_SOURCES.issubset(CONNECTOR_REGISTRY)


def test_expanded_stage1_connector_registry_contains_hpa() -> None:
    assert EXPANDED_STAGE1_SOURCES.issubset(CONNECTOR_REGISTRY)


def test_connectors_write_traceable_placeholder_outputs_without_network(tmp_path: Path) -> None:
    for source_id in sorted(MILESTONE2_SOURCES):
        result = CONNECTOR_REGISTRY[source_id](output_root=str(tmp_path), allow_network=False).run(retmax=1)

        assert result.source_id == source_id
        assert result.status == "partially_succeeded"
        assert result.used_sample_placeholder is True
        assert result.raw_document_count == 1
        assert result.raw_record_count == 1
        assert result.parsed_record_count == 1
        assert result.source_version
        assert result.source_checksum
        assert result.errors

        raw_document = result.raw_documents[0]
        assert raw_document.source_id == source_id
        assert raw_document.source_url
        assert raw_document.download_timestamp
        assert raw_document.raw_checksum
        assert raw_document.is_sample_placeholder is True
        assert Path(raw_document.raw_storage_path).exists()
        assert Path(raw_document.raw_metadata_path).exists()

        parsed_records = _json_lines(result.parsed_records_path)
        assert len(parsed_records) == 1
        parsed_record = parsed_records[0]
        assert parsed_record["source_id"] == source_id
        assert parsed_record["raw_document_path"] == raw_document.raw_storage_path
        assert parsed_record["raw_checksum"] == raw_document.raw_checksum
        assert parsed_record["source_location"] == "sample_placeholder"
        assert parsed_record["is_sample_placeholder"] is True

        log_payload = json.loads(Path(result.ingestion_log_path).read_text(encoding="utf-8"))
        assert log_payload["latest_run_metrics"]["parsed_record_count"] == 1
        assert log_payload["source_version"] == result.source_version
        assert log_payload["source_checksum"] == result.source_checksum
        assert log_payload["errors"][0]["retryable"] is True


def test_stage_a_connectors_write_traceable_placeholder_outputs_without_network(tmp_path: Path) -> None:
    for source_id in sorted(STAGE_A_SOURCES):
        result = CONNECTOR_REGISTRY[source_id](output_root=str(tmp_path), allow_network=False).run(retmax=1)

        assert result.source_id == source_id
        assert result.status == "partially_succeeded"
        assert result.used_sample_placeholder is True
        assert result.raw_document_count == 1
        assert result.raw_record_count == 1
        assert result.parsed_record_count == 1
        assert result.source_version
        assert result.source_checksum

        raw_document = result.raw_documents[0]
        assert raw_document.raw_checksum
        assert raw_document.source_url
        assert raw_document.is_sample_placeholder is True
        parsed_records = _json_lines(result.parsed_records_path)
        assert parsed_records[0]["source_id"] == source_id
        assert parsed_records[0]["source_location"] == "sample_placeholder"
        assert parsed_records[0]["raw_checksum"] == raw_document.raw_checksum

        log_payload = json.loads(Path(result.ingestion_log_path).read_text(encoding="utf-8"))
        assert log_payload["source_version"] == result.source_version
        assert log_payload["source_checksum"] == result.source_checksum


def test_hpa_connector_write_traceable_placeholder_output_without_network(tmp_path: Path) -> None:
    result = CONNECTOR_REGISTRY["hpa"](output_root=str(tmp_path), allow_network=False).run(retmax=1)

    assert result.source_id == "hpa"
    assert result.status == "partially_succeeded"
    assert result.used_sample_placeholder is True
    assert result.raw_document_count == 1
    assert result.parsed_record_count == 1
    assert result.source_version
    assert result.source_checksum


def test_hpa_connector_uses_pbpk_target_genes_for_compound_queries() -> None:
    assert HumanProteinAtlasConnector._genes_for_query("midazolam", 2) == ["CYP3A4", "CYP2C9"]
    assert HumanProteinAtlasConnector._genes_for_query("CYP3A4", 2) == ["CYP3A4"]


def test_connector_reruns_create_versioned_raw_snapshots(tmp_path: Path) -> None:
    connector_class = CONNECTOR_REGISTRY["pubchem"]

    first = connector_class(output_root=str(tmp_path), allow_network=False).run(retmax=1)
    second = connector_class(output_root=str(tmp_path), allow_network=False).run(retmax=1)

    first_raw = first.raw_documents[0].raw_storage_path
    second_raw = second.raw_documents[0].raw_storage_path
    assert first.ingestion_run_id != second.ingestion_run_id
    assert first_raw != second_raw
    assert Path(first_raw).exists()
    assert Path(second_raw).exists()


def test_run_connectors_cli_helper_runs_selected_sources_without_network(tmp_path: Path) -> None:
    results = run_sources(
        sources=["pubmed", "dailymed"],
        query="midazolam",
        retmax=1,
        output_root=str(tmp_path),
        allow_network=False,
    )

    assert [result["source_id"] for result in results] == ["pubmed", "dailymed"]
    assert all(result["parsed_record_count"] == 1 for result in results)


def test_run_connectors_source_registry_sync_can_be_mocked_without_db(tmp_path: Path, monkeypatch) -> None:
    calls = {}

    def fake_sync(results: list[dict], *, database_url: str | None = None) -> list[dict]:
        calls["database_url"] = database_url
        calls["source_ids"] = [result["source_id"] for result in results]
        return [
            {
                "source_id": result["source_id"],
                "source_version": result["source_version"],
                "source_checksum": result["source_checksum"],
            }
            for result in results
        ]

    monkeypatch.setattr("pbpk_kb.connectors.run_connectors.sync_source_registry_from_connector_results", fake_sync)
    results = run_sources(
        sources=["pubchem"],
        query="midazolam",
        retmax=1,
        output_root=str(tmp_path),
        allow_network=False,
        sync_source_registry=True,
        database_url="postgresql+psycopg://example",
    )

    assert calls == {"database_url": "postgresql+psycopg://example", "source_ids": ["pubchem"]}
    assert results[0]["source_registry_sync"]["source_id"] == "pubchem"


def test_dailymed_spl_xml_parser_extracts_day3_fields(tmp_path: Path) -> None:
    xml_payload = b"""<?xml version="1.0" encoding="UTF-8"?>
    <document xmlns="urn:hl7-org:v3">
      <component>
        <structuredBody>
          <component>
            <section>
              <code code="48780-1" displayName="SPL PRODUCT DATA ELEMENTS SECTION"/>
              <subject>
                <manufacturedProduct>
                  <manufacturedProduct>
                    <ingredient classCode="ACTIM">
                      <quantity>
                        <numerator value="2" unit="mg"/>
                        <denominator value="1" unit="mL"/>
                      </quantity>
                      <ingredientSubstance>
                        <code code="W7TTW573JJ"/>
                        <name>Midazolam Hydrochloride</name>
                        <activeMoiety>
                          <activeMoiety>
                            <code code="R60L0SM5BC"/>
                            <name>Midazolam</name>
                          </activeMoiety>
                        </activeMoiety>
                      </ingredientSubstance>
                    </ingredient>
                    <ingredient classCode="IACT">
                      <ingredientSubstance><name>Water</name></ingredientSubstance>
                    </ingredient>
                  </manufacturedProduct>
                </manufacturedProduct>
              </subject>
            </section>
          </component>
          <component>
            <section>
              <code code="34090-1" displayName="CLINICAL PHARMACOLOGY SECTION"/>
              <title>CLINICAL PHARMACOLOGY</title>
              <text>Clinical pharmacology text.</text>
            </section>
          </component>
          <component>
            <section>
              <code code="43682-4" displayName="PHARMACOKINETICS SECTION"/>
              <title>Pharmacokinetics</title>
              <text>Absorption and elimination text.</text>
            </section>
          </component>
        </structuredBody>
      </component>
    </document>
    """

    connector = DailyMedConnector(output_root=str(tmp_path), allow_network=False)
    parsed = connector.parse_spl_xml(xml_payload)

    assert parsed["active_ingredients"] == [
        {
            "name": "Midazolam Hydrochloride",
            "class_code": "ACTIM",
            "unii": "W7TTW573JJ",
            "strength_numerator_value": "2",
            "strength_numerator_unit": "mg",
            "strength_denominator_value": "1",
            "strength_denominator_unit": "mL",
        }
    ]
    assert parsed["active_moieties"] == [{"name": "Midazolam", "unii": "R60L0SM5BC"}]
    assert parsed["clinical_pharmacology_section"]["code"] == "34090-1"
    assert parsed["clinical_pharmacology_section"]["title"] == "CLINICAL PHARMACOLOGY"
    assert parsed["pk_section_candidates"][0]["code"] == "43682-4"
