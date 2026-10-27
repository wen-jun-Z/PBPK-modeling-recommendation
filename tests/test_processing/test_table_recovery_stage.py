from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from pbpk_kb.parsers.table_structure import parse_html_tables, parse_xml_table_fragment
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
from pbpk_kb.processing.table_recovery_stage import (
    TableRecoveryStage,
    central_estimate_count,
    detect_parameter,
    is_pdf_table_row,
    is_repeated_spanning_row,
    table_evidence_canonical_rank,
    table_evidence_fingerprint,
    table_structure_reliable,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row))
            file.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_xml_and_html_tables_expand_rowspan_and_colspan() -> None:
    fragment = """
    <table>
      <caption>PK summary</caption>
      <thead><tr><th rowspan="2">Parameter</th><th colspan="2">Treatment</th></tr>
      <tr><th>Low</th><th>High</th></tr></thead>
      <tbody><tr><td>Cmax (ng/mL)</td><td>10</td><td>20</td></tr></tbody>
    </table>
    """
    tables = parse_xml_table_fragment(fragment)
    assert len(tables) == 1
    assert tables[0].caption == "PK summary"
    assert tables[0].columns == ["Parameter", "Treatment / Low", "Treatment / High"]
    assert tables[0].rows == [["Cmax (ng/mL)", "10", "20"]]
    assert tables[0].metadata["rowspan_colspan_expanded"] is True

    html_tables = parse_html_tables(fragment)
    assert len(html_tables) == 1
    assert html_tables[0].columns == tables[0].columns
    assert html_tables[0].rows == tables[0].rows


def test_table_quality_gates_reject_misaligned_and_multi_estimate_cells() -> None:
    assert table_structure_reliable(["Parameter", "Value"], [["Cmax", "12 ng/mL"]]) is True
    assert table_structure_reliable(["footnote only"], [["group", "1.86", "3.20", "384"]]) is False
    assert central_estimate_count("0.72 ± 0.44") == 1
    assert central_estimate_count("288 (n=10)") == 1
    assert central_estimate_count("1.86 (±0.52) 4.12 (±1.83) 3.93 (±0.92)") == 3
    assert detect_parameter("Moderate renal impairment (CL cr 31-60 mL/min)") is None
    assert detect_parameter("Renal clearance (mL/min)") == "clearance"
    note = "* Values were measured below 20 mcg/mL and may change at higher concentrations."
    assert is_repeated_spanning_row([note, note, note]) is True
    assert is_repeated_spanning_row(["Elderly (>60 years)"] * 4) is True
    assert is_repeated_spanning_row(["10", "10", "10"]) is False
    assert is_pdf_table_row({"storage_path": "/raw/report.pdf", "source_location": "pmc:report.pdf#page=2"}) is True


def test_xml_table_promotes_spanning_title_and_td_header() -> None:
    tables = parse_xml_table_fragment(
        """<table><tbody>
        <tr><td colspan="4">Table 1: Metformin Pharmacokinetic Parameters</td></tr>
        <tr><td>Group</td><td>Cmax (mcg/mL)</td><td>Tmax (h)</td><td>Renal Clearance (mL/min)</td></tr>
        <tr><td>Moderate impairment</td><td>4.12</td><td>3.75</td><td>108</td></tr>
        </tbody></table>"""
    )
    assert len(tables) == 1
    assert tables[0].caption == "Table 1: Metformin Pharmacokinetic Parameters"
    assert tables[0].columns == ["Group", "Cmax (mcg/mL)", "Tmax (h)", "Renal Clearance (mL/min)"]
    assert tables[0].rows == [["Moderate impairment", "4.12", "3.75", "108"]]


def test_table_evidence_fingerprint_deduplicates_mirrors_and_prefers_dailymed() -> None:
    base = {
        "evidence_id": "daily",
        "source_id": "dailymed",
        "compound": "metformin",
        "parameter": "Tmax",
        "canonical_value": 2.75,
        "canonical_unit": "h",
        "source_text": "500 mg single dose | 1.03 ± 0.33 | 2.75 ± 0.81",
        "condition": "Dose: 500 mg",
        "source_location": "dailymed:label#table=1#row=2",
    }
    mirror = {
        **base,
        "evidence_id": "openfda",
        "source_id": "openfda_drug_label",
        "condition": "Dose 500 mg",
        "source_location": "openfda:label#table=1#row=2",
    }
    assert table_evidence_fingerprint(base) == table_evidence_fingerprint(mirror)
    assert table_evidence_canonical_rank(base) < table_evidence_canonical_rank(mirror)


def test_pdf_coordinate_table_never_creates_deterministic_evidence(tmp_path: Path) -> None:
    stage = TableRecoveryStage(output_dir=tmp_path)
    stage.target_compounds = {"midazolam"}
    row = {
        "table_row_id": "row",
        "source_id": "pmc_oa",
        "caption": "Midazolam pharmacokinetics",
        "columns_json": ["Parameter", "Value"],
        "cells_json": ["CL/F (L/h)", "0.6"],
        "row_text": "CL/F (L/h) | 0.6",
        "metadata_json": {"structure_reliable": True, "table_origin": "recovered_pdf_table"},
    }
    assert stage._extract_table_evidence(row, (None, "asset_level_ownership_unresolved")) == []


def test_table_recovery_reuses_dailymed_recovers_openfda_and_rejects_pubmed_pseudo_table(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed" / "run_test"
    layout = raw_processing_layout(output_dir)
    layout.ensure_dirs()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    daily_path = raw_dir / "daily.xml"
    daily_path.write_text(
        """<document xmlns="urn:hl7-org:v3">
        <component><structuredBody><component><section><title>Pharmacokinetics</title><text>
        <table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody><tr><td>Cmax</td><td>12 ng/mL</td></tr></tbody></table>
        </text></section></component></structuredBody></component>
        <subject><manufacturedProduct><manufacturedProduct><ingredient classCode="ACTIM">
        <quantity><numerator value="1" unit="mg"/><denominator value="1" unit="mL"/></quantity>
        <ingredientSubstance><code code="R60L0SM5BC"/><name>Midazolam Hydrochloride</name></ingredientSubstance>
        </ingredient></manufacturedProduct></manufacturedProduct></subject>
        </document>""",
        encoding="utf-8",
    )
    openfda_path = raw_dir / "openfda.json"
    openfda_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "set_id": "unmatched-set",
                        "pharmacokinetics": ["Caffeine pharmacokinetic values are shown in the table."],
                        "pharmacokinetics_table": [
                            "<table><caption>Caffeine PK</caption><thead><tr><th>Parameter</th><th>Value</th></tr></thead>"
                            "<tbody><tr><td>AUC</td><td>48 ng*h/mL</td></tr></tbody></table>"
                        ],
                        "openfda": {"substance_name": ["CAFFEINE"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pubmed_path = raw_dir / "pubmed.xml"
    pubmed_path.write_text(
        "<PubmedArticleSet><PubmedArticle><AbstractText>Warfarin values 1 2 3 4 5 were reported.</AbstractText></PubmedArticle></PubmedArticleSet>",
        encoding="utf-8",
    )

    def inventory(asset_id: str, source_id: str, compound: str, path: Path, family: str) -> dict:
        payload = path.read_bytes()
        return {
            "run_id": "run_test",
            "raw_asset_id": asset_id,
            "source_id": source_id,
            "compound": compound,
            "storage_path": str(path),
            "source_url": f"https://example.test/{path.name}",
            "checksum": sha256(payload).hexdigest(),
            "file_family": family,
            "file_extension": path.suffix,
            "asset_metadata_json": {},
        }

    daily = inventory("daily", "dailymed", "midazolam", daily_path, "xml")
    openfda = inventory("openfda", "openfda_drug_label", "caffeine", openfda_path, "json")
    pubmed = inventory("pubmed", "pubmed", "warfarin", pubmed_path, "xml")
    write_jsonl(layout.path("raw_inventory.jsonl", for_write=True), [daily, openfda, pubmed])
    write_jsonl(
        layout.path("parsed_tables.jsonl", for_write=True),
        [
            {
                "run_id": "run_test",
                "parsed_table_id": "daily-table",
                "raw_asset_id": "daily",
                "source_id": "dailymed",
                "compound": "midazolam",
                "storage_path": str(daily_path),
                "source_url": daily["source_url"],
                "checksum": daily["checksum"],
                "source_location": "dailymed:daily.xml#xml_path=//section[1]#table=1",
                "columns_json": ["Parameter", "Value"],
                "cells_json": [["Cmax", "12 ng/mL"]],
                "table_text": "Cmax | 12 ng/mL",
                "metadata_json": {"section_title": "Pharmacokinetics"},
            }
        ],
    )

    common = {
        "run_id": "run_test",
        "processing_status": "deferred_unparsed_flattened_table",
        "parsed_document_id": "document",
    }
    statuses = [
        {
            **common,
            "chunk_id": "daily-chunk",
            "raw_asset_id": "daily",
            "source_id": "dailymed",
            "compound": "midazolam",
            "storage_path": str(daily_path),
            "source_url": daily["source_url"],
            "checksum": daily["checksum"],
            "source_location": "dailymed:daily.xml#xml_path=//section[1]#chunk=1",
        },
        {
            **common,
            "chunk_id": "openfda-chunk",
            "raw_asset_id": "openfda",
            "source_id": "openfda_drug_label",
            "compound": "caffeine",
            "storage_path": str(openfda_path),
            "source_url": openfda["source_url"],
            "checksum": openfda["checksum"],
            "source_location": "openfda_drug_label:openfda.json#json_path=$.results[0].pharmacokinetics#chunk=1",
        },
        {
            **common,
            "chunk_id": "pubmed-chunk",
            "raw_asset_id": "pubmed",
            "source_id": "pubmed",
            "compound": "warfarin",
            "storage_path": str(pubmed_path),
            "source_url": pubmed["source_url"],
            "checksum": pubmed["checksum"],
            "source_location": "pubmed:pubmed.xml#xml_path=//PubmedArticle[1]/Abstract#chunk=1",
        },
    ]
    write_jsonl(layout.path("chunk_processing_status.jsonl", for_write=True), statuses)
    write_jsonl(layout.path("evidence_candidates_deterministic.jsonl", for_write=True), [])
    write_jsonl(layout.path("evidence_candidates_llm.jsonl", for_write=True), [])
    write_json(
        layout.path("pipeline_state.json", for_write=True),
        {
            "run_id": "run_test",
            "current_stage": "llm_extract",
            "stages": {
                "llm_extract": {"status": "completed", "accepted": True},
                "completeness": {"status": "not_started"},
                "embedding": {"status": "not_started"},
            },
        },
    )

    summary = TableRecoveryStage(output_dir=output_dir).run()

    assert summary["accepted"] is True
    assert summary["deferred_chunk_count"] == 3
    assert summary["terminal_status_counts"] == {
        "not_table_after_source_inspection": 1,
        "recovered_from_raw": 1,
        "resolved_by_existing_table": 1,
    }
    evidence = read_jsonl(layout.path("table_evidence_candidates.jsonl"))
    assert {(row["compound"], row["parameter"], row["value"], row["unit"]) for row in evidence} == {
        ("midazolam", "Cmax", 12.0, "ng/mL"),
        ("caffeine", "AUC", 48.0, "ng*h/mL"),
    }
    assert all(row["curation_status"] == "needs_review" for row in evidence)
    assert all(row["source_text"] and row["source_location"] and row["checksum"] for row in evidence)
    statuses = read_jsonl(layout.path("table_recovery_status.jsonl"))
    pubmed_status = next(row for row in statuses if row["chunk_id"] == "pubmed-chunk")
    assert pubmed_status["terminal_status"] == "not_table_after_source_inspection"
    assert "does not contain full article table bodies" in pubmed_status["reason"]
    state = json.loads(layout.path("pipeline_state.json").read_text(encoding="utf-8"))
    assert state["stages"]["table_recovery"]["accepted"] is True
