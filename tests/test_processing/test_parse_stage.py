from __future__ import annotations

import json
import sqlite3
import zipfile
from datetime import time
from hashlib import sha256
from pathlib import Path

from openpyxl import Workbook

from pbpk_kb.parsers.raw_formats import pdf_table_likelihood
from pbpk_kb.parsers.emitter import PARSER_VERSION
from pbpk_kb.parsers.models import ParseAsset
from pbpk_kb.processing.parse_stage import (
    RawParseStage,
    archive_path_rejection,
    parse_bsdtar_verbose_listing,
    resolve_asset_links,
)
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout


def write_inventory(output_dir: Path, assets: list[dict]) -> None:
    layout = raw_processing_layout(output_dir)
    layout.ensure_dirs()
    inventory_path = layout.path("raw_inventory.jsonl", for_write=True)
    inventory_path.write_text("".join(json.dumps(asset) + "\n" for asset in assets), encoding="utf-8")
    layout.path("inventory_summary.json", for_write=True).write_text(
        json.dumps({"stage": "inventory", "status": "completed", "accepted": True, "raw_asset_count": len(assets)}),
        encoding="utf-8",
    )
    layout.path("pipeline_state.json", for_write=True).write_text(
        json.dumps(
            {
                "run_id": output_dir.name,
                "stages": {
                    "inventory": {"status": "completed", "accepted": True},
                    "parse": {"status": "not_started"},
                },
            }
        ),
        encoding="utf-8",
    )


def inventory_asset(path: Path, *, asset_id: str, source_id: str, family: str, compound: str = "midazolam") -> dict:
    payload = path.read_bytes()
    return {
        "run_id": "run_test",
        "raw_asset_id": asset_id,
        "source_id": source_id,
        "compound": compound,
        "group": "test",
        "storage_path": str(path),
        "source_url": f"https://example.test/{path.name}",
        "downloaded_at": "2026-07-15T00:00:00Z",
        "checksum": sha256(payload).hexdigest(),
        "content_type": None,
        "file_extension": path.suffix.lower(),
        "file_family": family,
        "file_size": len(payload),
        "request_id": asset_id,
        "request_key": "test",
        "asset_metadata_json": {},
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_duplicate_invalid_xml_keeps_each_failure_and_valid_duplicates_still_link(tmp_path: Path) -> None:
    raw = tmp_path / "run_test"
    raw.mkdir()
    assets = []
    for index in range(3):
        path = raw / f"bad_label_{index}.xml"
        path.write_bytes(b"None")
        assets.append(inventory_asset(path, asset_id=f"bad:{index}", source_id="dailymed", family="xml"))
    for index in range(2):
        path = raw / f"valid_{index}.json"
        path.write_text('{"compound": "midazolam", "value": 1}')
        assets.append(inventory_asset(path, asset_id=f"valid:{index}", source_id="pubchem", family="json"))
    output = tmp_path / "processed"
    write_inventory(output, assets)
    summary = RawParseStage(run_root=raw, output_dir=output).run()
    layout = raw_processing_layout(output)

    assert summary["accepted"] is False and summary["status"] == "failed"
    assert summary["failed_asset_count"] == 3
    assert summary["processing_failure_count"] == 3
    assert summary["alias_asset_count"] == 1
    failures = read_jsonl(layout.path("processing_failures.jsonl"))
    assert {row["raw_asset_id"] for row in failures} == {f"bad:{index}" for index in range(3)}
    statuses = {row["raw_asset_id"]: row["terminal_status"] for row in read_jsonl(layout.path("asset_parse_status.jsonl"))}
    assert all(statuses[f"bad:{index}"] == "failed" for index in range(3))
    assert statuses["valid:0"] == "parsed" and statuses["valid:1"] == "duplicate_linked"
    aliases = read_jsonl(layout.path("raw_asset_aliases.jsonl"))
    assert len(aliases) == 1
    assert aliases[0]["alias_raw_asset_id"] == "valid:1"
    assert aliases[0]["canonical_raw_asset_id"] == "valid:0"


def test_provider_outage_receipt_never_becomes_parsed_content(tmp_path: Path) -> None:
    raw = tmp_path / "run_test"
    raw.mkdir()
    response = raw / "response.bin"
    response.write_bytes(b"<html>HTTP 500; quinidine clearance 99 L/h</html>")
    asset = inventory_asset(response, asset_id="outage", source_id="chembl", family="html", compound="quinidine")
    asset["asset_metadata_json"] = {"processing_policy": "metadata_only",
                                    "processing_policy_reason": "official_chembl_service_unavailable"}
    output = tmp_path / "processed"
    write_inventory(output, [asset])
    result = RawParseStage(run_root=raw, output_dir=output).run()
    layout = raw_processing_layout(output)
    assert result["accepted"] is True
    assert read_jsonl(layout.path("asset_parse_status.jsonl"))[0]["terminal_status"] == "metadata_only"
    for name in ("parsed_documents.jsonl", "parsed_records.jsonl", "text_chunks.jsonl"):
        path = layout.path(name)
        assert not path.exists() or read_jsonl(path) == []


def test_html_office_suffix_and_archive_sidecars_are_dispatched_by_content(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    html = raw / "saved_page.docx"
    html.write_text('<!DOCTYPE html><html><body><p>Observed drug exposure was measured in healthy adults.</p></body></html>')
    archive = raw / "supplements.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("__MACOSX/._results.xlsx", bytes.fromhex("0005160700020000") + b"\x00" * 100)
        z.writestr("folder/~$results.docx", b"\x04USER" + b"\x00" * 100)
        z.writestr("results.csv", "time,concentration\n1,2\n2,3\n")
    output = tmp_path / "processed"
    write_inventory(output, [inventory_asset(html, asset_id="html", source_id="pmc_oa", family="document"),
                             inventory_asset(archive, asset_id="zip", source_id="pmc_oa", family="archive")])
    result = RawParseStage(run_root=raw, output_dir=output).run()
    assert result["accepted"] is True
    layout = raw_processing_layout(output)
    documents = read_jsonl(layout.path("parsed_documents.jsonl"))
    assert any("Observed drug exposure" in d["text"] for d in documents)
    members = {d["inner_path"]: d for d in read_jsonl(layout.path("archive_members.jsonl"))}
    assert members["__MACOSX/._results.xlsx"]["terminal_status"] == "metadata_only"
    assert members["folder/~$results.docx"]["terminal_status"] == "metadata_only"
    assert members["results.csv"]["terminal_status"] == "parsed"
    tables = read_jsonl(layout.path("parsed_tables.jsonl"))
    assert len(tables) == 1


def test_metadata_like_names_do_not_hide_real_documents_and_ole_precedes_embedded_zip(tmp_path: Path) -> None:
    from pbpk_kb.processing.parse_stage import prepare_asset_for_dispatch
    html = tmp_path / "~$actual.docx"
    html.write_text('<html><body>Actual article content</body></html>')
    asset = ParseAsset.from_inventory(inventory_asset(html, asset_id="real", source_id="pmc_oa", family="document"))
    assert prepare_asset_for_dispatch(asset)[0].file_family == "html"
    ole = tmp_path / "legacy.docx"
    ole.write_bytes(bytes.fromhex("d0cf11e0a1b11ae1") + b"\x00" * 512)
    with zipfile.ZipFile(ole, "a") as z:
        z.writestr("[Content_Types].xml", '<Types><Override ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/></Types>')
    asset = ParseAsset.from_inventory(inventory_asset(ole, asset_id="ole", source_id="pmc_oa", family="document"))
    result, _, metadata = prepare_asset_for_dispatch(asset)
    assert result.file_family == "document" and result.file_extension == ".doc"
    assert metadata["detected_format"] == "legacy_word_ole"
    assert result.storage_path == asset.storage_path and result.checksum == asset.checksum


def test_webp_magic_and_image_metadata_preserve_source_without_evidence(tmp_path: Path) -> None:
    from PIL import Image
    from pbpk_kb.parsers.file_types import file_family_for, sniff_bytes_family

    path = tmp_path / "figure.bin"
    Image.new("RGB", (16, 12), (30, 80, 120)).save(path, format="WEBP")
    assert file_family_for(path, ".bin", None, "pmc_oa") == "image"
    assert file_family_for(path, ".webp", None, "pmc_oa") == "image"
    assert sniff_bytes_family(b"RIFF\x00\x00\x00\x00WAVE") == "media"
    assert sniff_bytes_family(b"RIFF\x00\x00\x00\x00AVI ") == "media"
    output = tmp_path / "processed/run_test"
    write_inventory(output, [inventory_asset(path, asset_id="asset:webp", source_id="pmc_oa", family="image")])
    summary = RawParseStage(run_root=tmp_path / "run_test", output_dir=output).run()
    assert summary["accepted"] is True
    assert summary["evidence_rows_created"] == 0
    layout = raw_processing_layout(output)
    records = read_jsonl(layout.path("parsed_records.jsonl"))
    assert records[0]["raw_asset_id"] == "asset:webp"
    assert records[0]["checksum"] == sha256(path.read_bytes()).hexdigest()
    assert records[0]["record_json"]["format"] == "WEBP"
    assert records[0]["record_json"]["width"] == 16


def test_parse_stage_dispatches_formats_preserves_provenance_and_links_duplicates(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output_dir = tmp_path / "processed" / "run_test"

    json_path = raw_dir / "pubchem.json"
    json_path.write_text(json.dumps({"CID": 123, "MolecularWeight": 325.8}), encoding="utf-8")
    duplicate_path = raw_dir / "pubchem_duplicate.json"
    duplicate_path.write_bytes(json_path.read_bytes())
    tolerant_json_path = raw_dir / "pubmed_control_character.json"
    tolerant_json_path.write_bytes(b'{"record":"source text with control \x01 character"}')

    pubmed_path = raw_dir / "pubmed.xml"
    pubmed_path.write_text(
        """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID><Article><ArticleTitle>PK study</ArticleTitle><Abstract><AbstractText>Midazolam pharmacokinetics were evaluated in healthy adults after an oral dose.</AbstractText></Abstract><Journal><Title>Clinical PK</Title><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/test</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>""",
        encoding="utf-8",
    )

    sqlite_path = raw_dir / "model.db"
    connection = sqlite3.connect(sqlite_path)
    connection.execute("CREATE TABLE parameters (name TEXT, value REAL, description TEXT)")
    connection.execute(
        "INSERT INTO parameters VALUES (?, ?, ?)",
        ("clearance", 12.5, "Systemic clearance measured after intravenous administration in healthy adults."),
    )
    connection.commit()
    connection.close()

    workbook_path = raw_dir / "table.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "PK"
    sheet.append(["parameter", "value", "unit"])
    sheet.append(["Cmax", 10.2, "ng/mL"])
    sheet.append(["sampling_time", time(8, 30), None])
    workbook.save(workbook_path)

    mislabeled_workbook_path = raw_dir / "supplement.docx"
    mislabeled_workbook = Workbook()
    mislabeled_sheet = mislabeled_workbook.active
    mislabeled_sheet.append(["parameter", "value", "unit"])
    mislabeled_sheet.append(["AUC", 48.1, "ng*h/mL"])
    mislabeled_workbook.save(mislabeled_workbook_path)

    pdf_path = raw_dir / "report.pdf"
    import fitz

    pdf = fitz.open()
    for page_number in [1, 2]:
        page = pdf.new_page()
        page.insert_text((72, 72), f"Midazolam pharmacokinetic report page {page_number} with sufficient text.")
    pdf.save(pdf_path)
    pdf.close()

    archive_path = raw_dir / "package.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("duplicate.json", json_path.read_bytes())
        archive.writestr("notes.txt", "A supplementary pharmacokinetic note with traceable archive provenance.")

    assets = [
        inventory_asset(json_path, asset_id="asset:json", source_id="pubchem", family="json"),
        inventory_asset(duplicate_path, asset_id="asset:duplicate", source_id="pubchem", family="json", compound="warfarin"),
        inventory_asset(tolerant_json_path, asset_id="asset:tolerant-json", source_id="pubmed", family="json"),
        inventory_asset(pubmed_path, asset_id="asset:pubmed", source_id="pubmed", family="xml"),
        inventory_asset(sqlite_path, asset_id="asset:sqlite", source_id="pmc_oa", family="database"),
        inventory_asset(workbook_path, asset_id="asset:xlsx", source_id="pmc_oa", family="table"),
        inventory_asset(
            mislabeled_workbook_path,
            asset_id="asset:mislabeled-xlsx",
            source_id="pmc_oa",
            family="document",
        ),
        inventory_asset(pdf_path, asset_id="asset:pdf", source_id="pmc_oa", family="pdf"),
        inventory_asset(archive_path, asset_id="asset:archive", source_id="pmc_oa", family="archive"),
    ]
    write_inventory(output_dir, assets)

    summary = RawParseStage(run_root=tmp_path / "raw_run" / "run_test", output_dir=output_dir).run()

    assert summary["accepted"] is True
    assert summary["asset_parse_status_count"] == len(assets)
    assert summary["alias_asset_count"] == 1
    assert summary["failed_asset_count"] == 0
    assert summary["processing_failure_count"] == 0
    assert summary["embeddings_created"] == 0
    assert summary["evidence_rows_created"] == 0

    layout = raw_processing_layout(output_dir)
    statuses = read_jsonl(layout.path("asset_parse_status.jsonl"))
    assert len(statuses) == len(assets)
    assert len({row["raw_asset_id"] for row in statuses}) == len(assets)
    assert any(row["terminal_status"] == "duplicate_linked" for row in statuses)
    tolerant_status = next(row for row in statuses if row["raw_asset_id"] == "asset:tolerant-json")
    assert tolerant_status["terminal_status"] == "parsed"
    assert tolerant_status["metadata_json"]["tolerant_json_parse"] is True
    assert tolerant_status["warnings"]
    mislabeled_status = next(row for row in statuses if row["raw_asset_id"] == "asset:mislabeled-xlsx")
    assert mislabeled_status["terminal_status"] == "parsed"
    assert mislabeled_status["parser_name"] == "table_parser"
    assert mislabeled_status["metadata_json"]["declared_file_family"] == "document"
    assert mislabeled_status["metadata_json"]["detected_file_family"] == "table"
    assert mislabeled_status["metadata_json"]["ooxml_type_mismatch"] is True
    assert mislabeled_status["warnings"]

    aliases = read_jsonl(layout.path("raw_asset_aliases.jsonl"))
    assert aliases == [
        {
            "alias_compound": "warfarin",
            "alias_raw_asset_id": "asset:duplicate",
            "alias_request_id": "asset:duplicate",
            "alias_request_key": "test",
            "alias_source_id": "pubchem",
            "alias_source_url": "https://example.test/pubchem_duplicate.json",
            "alias_storage_path": str(duplicate_path),
            "canonical_raw_asset_id": "asset:json",
            "checksum": assets[0]["checksum"],
            "file_family": "json",
            "parser_name": "json_parser",
            "parser_version": PARSER_VERSION,
            "run_id": "run_test",
        }
    ]

    archive_members = read_jsonl(layout.path("archive_members.jsonl"))
    duplicate_member = next(row for row in archive_members if row["inner_path"] == "duplicate.json")
    assert duplicate_member["terminal_status"] == "duplicate_linked"
    assert duplicate_member["duplicate_target_raw_asset_id"] == "asset:json"

    documents = read_jsonl(layout.path("parsed_documents.jsonl"))
    assert len([row for row in documents if row["content_kind"] == "pdf_page"]) == 2
    assert all(row["raw_asset_id"] and row["checksum"] and row["source_location"] for row in documents)
    assert read_jsonl(layout.path("parsed_records.jsonl"))
    parsed_tables = read_jsonl(layout.path("parsed_tables.jsonl"))
    assert parsed_tables
    assert any("08:30:00" in str(row["cells_json"]) for row in parsed_tables)
    progress = json.loads(layout.path("parse_progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"


def test_archive_path_rejection_blocks_traversal_and_absolute_paths() -> None:
    assert archive_path_rejection("../escape.txt") == "path_traversal"
    assert archive_path_rejection("/absolute.txt") == "absolute_path"
    assert archive_path_rejection("safe/nested/file.txt") is None


def test_bsdtar_listing_preserves_member_names_with_spaces() -> None:
    rows = parse_bsdtar_verbose_listing(
        "-rw-r--r--  0 0 0 90818 Dec 19 2018 cells-422282 supp/Fig S1.pptx\n"
        "drwxr-xr-x  0 0 0 0 Jan 11 2019 cells-422282 supp\n"
    )
    assert rows == [
        {"mode": "-rw-r--r--", "size": 90818, "inner_path": "cells-422282 supp/Fig S1.pptx"},
        {"mode": "drwxr-xr-x", "size": 0, "inner_path": "cells-422282 supp"},
    ]


def test_asset_links_resolve_identical_copies_to_canonical_asset(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "PMC123"
    raw_dir.mkdir(parents=True)
    canonical_path = raw_dir / "fig1_aaaaaaaaaaaa.jpg"
    alias_path = raw_dir / "fig1_bbbbbbbbbbbb.jpg"
    canonical_path.write_bytes(b"identical-image")
    alias_path.write_bytes(canonical_path.read_bytes())
    canonical = inventory_asset(canonical_path, asset_id="asset:canonical", source_id="pmc_oa", family="image")
    alias = inventory_asset(alias_path, asset_id="asset:alias", source_id="pmc_oa", family="image")
    output_dir = tmp_path / "processed" / "run_test"
    layout = raw_processing_layout(output_dir)
    layout.ensure_dirs()
    layout.path("raw_asset_aliases.jsonl", for_write=True).write_text(
        json.dumps(
            {
                "alias_raw_asset_id": "asset:alias",
                "canonical_raw_asset_id": "asset:canonical",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    layout.path("asset_links.jsonl", for_write=True).write_text(
        json.dumps(
            {
                "target_href": "fig1.jpg",
                "metadata_json": {"pmcid": "PMC123"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    counts = resolve_asset_links(
        layout,
        [ParseAsset.from_inventory(canonical), ParseAsset.from_inventory(alias)],
    )

    assert counts == {"total": 1, "resolved": 1, "unresolved": 0, "ambiguous": 0}
    link = read_jsonl(layout.path("asset_links.jsonl"))[0]
    assert link["resolution_status"] == "resolved"
    assert link["resolution_method"] == "identical_checksum_canonical"
    assert link["target_raw_asset_id"] == "asset:canonical"
    assert link["candidate_target_raw_asset_ids"] == ["asset:alias", "asset:canonical"]


def test_pdf_table_likelihood_skips_prose_and_detects_numeric_tables() -> None:
    assert pdf_table_likelihood("A normal pharmacokinetic paragraph without a table.", [()] * 2)[0] is False
    candidate, reasons = pdf_table_likelihood(
        "Parameter    Mean    SD\nCmax    10.2    1.4\nAUC    45.0    5.2\nTmax    2.0    0.5",
        [()] * 4,
    )
    assert candidate is True
    assert reasons
