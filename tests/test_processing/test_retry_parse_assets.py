from __future__ import annotations

import importlib.util
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pytest

from pbpk_kb.parsers.emitter import ParseEmitter
from pbpk_kb.parsers.models import ParseAsset, ParseOutcome
from pbpk_kb.parsers.raw_formats import RawFormatParsers
from pbpk_kb.parsers.registry import ParserRegistry
from pbpk_kb.processing.parse_stage import RawParseStage
from pbpk_kb.processing.raw_processing_paths import raw_processing_layout

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("retry_parse_assets_under_test", PROJECT / "scripts/deployment/retry_parse_assets.py")
retry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = retry
spec.loader.exec_module(retry)
RUN = "run_20261009T140923Z"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    raw = root / "artifacts/pbpk_raw_runs/compound_runs" / RUN
    raw.mkdir(parents=True)
    output = root / "data_process/raw_processing" / RUN
    layout = raw_processing_layout(output)
    layout.ensure_dirs()
    assets = []
    for label in ("good", "failed"):
        path = raw / (label + ".txt")
        path.write_text(label + " human-readable concentration evidence fixture for this parser, no trial data.")
        assets.append(ParseAsset(run_id=RUN, raw_asset_id=label, source_id="fixture", compound="example",
            group=None, storage_path=str(path), source_url=None, downloaded_at=None,
            checksum=retry.digest(path), content_type="text/plain", file_extension=".txt", file_family="text",
            file_size=path.stat().st_size, request_id=None, request_key=None, asset_metadata_json={}))
    with layout.path("raw_inventory.jsonl", for_write=True).open("w") as stream:
        for asset in assets:
            stream.write(json.dumps(asset.__dict__) + "\n")
    write(layout.path("inventory_summary.json", for_write=True), {"accepted": True, "status": "completed", "raw_asset_count": 2})
    registry = ParserRegistry()
    RawFormatParsers().register(registry)
    with ParseEmitter(layout, RUN) as emitter:
        good = assets[0]
        parser = registry.resolve(good)
        outcome = parser.callback(good, emitter)
        emitter.status(good, parser_name=parser.name, status=outcome.status)
        emitter.failure(assets[1], parser_name="text_parser", failure_type="parse_exception", error="original failure")
        emitter.status(assets[1], parser_name="text_parser", status="failed")
        summary = RawParseStage(run_root=raw, output_dir=output)._build_summary(emitter,
            expected_asset_count=2, processed_asset_count=2, canonical_asset_count=1, registry=registry)
    summary.update(accepted=False, status="failed")
    write(layout.path("parse_summary.json", for_write=True), summary)
    write(layout.path("parse_progress.json", for_write=True), {"status": "failed"})
    write(layout.path("pipeline_state.json", for_write=True), {"run_id": RUN, "current_stage": "parse", "stages": {
        "inventory": {"status": "completed", "accepted": True}, "parse": {"status": "failed", "accepted": False},
        "deterministic_extract": {"status": "not_started"}}})
    stages = {name: {"status": "completed", "accepted": True} for name in
              ("raw_acquisition", "raw_finalization", "pmc_static_transfer", "raw_reconciliation", "raw_inventory")}
    stages["parse"] = {"status": "failed", "accepted": False, "error": "original failure"}
    write(root / "data_process/pipeline_runs" / (RUN + ".json"), {"run_id": RUN, "status": "failed", "current_stage": "parse",
        "compounds": ["example"], "raw_run_root": str(raw), "raw_processing_dir": str(output), "stages": stages})
    identity = {"pid": 1999999998, "start_ticks": "123"}
    controller = {"pid": 1999999997, "start_ticks": "124"}
    write(root / retry.DEPLOY / "status.json", {"status": "stopped_for_review", "controller": controller, "frozen_inputs": {},
        "current_batch": {"run_id": RUN, "status": "running_pilot", "pipeline_pid": identity["pid"],
                          "pipeline_identity": identity, "watchdog_pid": 1999999996, "compounds": ["example"]}})
    write(root / "runtime/deployment/evidence_monitor" / RUN / "status.json", {"target": identity, "monitor_pid": 1999999996})
    for name in retry.CODE:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen source fixture")
    return root, layout, assets


def recovery(root):
    return root / retry.DEPLOY / "recovery_test"


def active_snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in (root / "data_process").rglob("*") if path.is_file()}


def test_success_preserves_other_bytes_and_rolls_back_exactly(project):
    root, layout, _ = project
    before = active_snapshot(root)
    receipt = retry.prepare(root, RUN, ["failed"], recovery(root))
    assert receipt["status"] == "prepared"
    assert active_snapshot(root) == before
    assert receipt["summary"]["accepted"] is True
    assert receipt["summary"]["asset_parse_status_count"] == 2
    assert receipt["summary"]["parsed_document_count"] == 2
    assert receipt["summary"]["canonical_asset_count"] == 2
    for name in retry.CODE:
        assert name in receipt["original"]
    retry.apply(root, recovery(root))
    after = active_snapshot(root)
    for name in ParseEmitter.OUTPUT_NAMES:
        if name not in retry.LEDGERS:
            key = str(layout.path(name).relative_to(root))
            assert after[key].startswith(before[key])
    assert layout.path("processing_failures.jsonl").read_bytes() == b""
    manifest = retry.read(root / "data_process/pipeline_runs" / (RUN + ".json"))
    assert manifest["status"] == "ready_to_resume"
    assert manifest["stages"]["parse"]["previous_failed_attempt"]["error"] == "original failure"
    assert manifest["stages"]["parse"]["accepted"] is True
    assert manifest["stages"]["parse"]["result"]["summary"]["accepted"] is True
    retry.apply(root, recovery(root), rollback=True)
    assert active_snapshot(root) == before


def test_reject_partial_output_even_if_status_claims_empty(project):
    root, layout, assets = project
    asset = assets[1]
    row = {"run_id": RUN, "raw_asset_id": "failed", "checksum": asset.checksum, "source_location": "partial"}
    with layout.path("parsed_records.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    before = active_snapshot(root)
    with pytest.raises(RuntimeError, match="Partial output"):
        retry.prepare(root, RUN, ["failed"], recovery(root))
    assert active_snapshot(root) == before


@pytest.mark.parametrize("field,value,match", [
    ("checksum", "0" * 64, "checksum differs"),
    ("raw_asset_id", "outside", "identity missing"),
    ("run_id", "run_other", "another run"),
])
def test_reject_wrong_output_identity(project, field, value, match):
    root, layout, _ = project
    path = layout.path("asset_parse_status.jsonl")
    data = [row for _, row in retry.rows(path)]
    data[1][field] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in data))
    with pytest.raises(RuntimeError, match=match):
        retry.prepare(root, RUN, ["failed"], recovery(root))


def test_reject_raw_path_outside_project(project):
    root, layout, _ = project
    path = layout.path("raw_inventory.jsonl")
    data = [row for _, row in retry.rows(path)]
    data[1]["storage_path"] = str(root.parent / "unrelated.txt")
    path.write_text("".join(json.dumps(row) + "\n" for row in data))
    with pytest.raises(RuntimeError, match="outside project"):
        retry.prepare(root, RUN, ["failed"], recovery(root))


def test_reject_recovery_outside_project(project):
    root, _, _ = project
    with pytest.raises(RuntimeError, match="outside project"):
        retry.prepare(root, RUN, ["failed"], root.parent / "unrelated")


def test_reject_changed_input_between_prepare_and_apply(project):
    root, layout, _ = project
    retry.prepare(root, RUN, ["failed"], recovery(root))
    path = layout.path("parse_progress.json")
    path.write_text('{"status":"concurrent"}\n')
    before = active_snapshot(root)
    with pytest.raises(RuntimeError, match="Concurrent or changed input"):
        retry.apply(root, recovery(root))
    assert active_snapshot(root) == before


def test_reject_changed_parser_after_prepare(project):
    root, _, _ = project
    retry.prepare(root, RUN, ["failed"], recovery(root))
    (root / "src/pbpk_kb/parsers/source_integrity.py").write_text("changed")
    with pytest.raises(RuntimeError, match="Concurrent or changed input"):
        retry.apply(root, recovery(root))


def test_reject_live_registered_process(project):
    import os
    root, _, _ = project
    path = root / retry.DEPLOY / "status.json"
    state = retry.read(path)
    fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
    state["controller"] = {"pid": os.getpid(), "start_ticks": fields[19]}
    write(path, state)
    with pytest.raises(RuntimeError, match="has not exited"):
        retry.prepare(root, RUN, ["failed"], recovery(root))


@pytest.mark.parametrize("verified", [False, True])
def test_unsupported_requires_explicit_verified_source_damage(project, monkeypatch, verified):
    root, _, _ = project
    original = RawFormatParsers.parse_text
    def outcome(self, asset, emitter):
        if asset.raw_asset_id != "failed":
            return original(self, asset, emitter)
        return ParseOutcome(status="unsupported", metadata={"source_content_unavailable": True,
            "checksum_verified": verified, "official_etag_verified": verified,
            "reason": "official_ole_directory_out_of_bounds"})
    monkeypatch.setattr(RawFormatParsers, "parse_text", outcome)
    if verified:
        receipt = retry.prepare(root, RUN, ["failed"], recovery(root))
        assert receipt["summary"]["unsupported_asset_count"] == 1
        assert receipt["summary"]["parsed_document_count"] == 1
        assert receipt["summary"]["source_content_unavailable_count"] == 1
        assert receipt["summary"]["source_content_unavailable_asset_ids"] == ["failed"]
        assert receipt["append"] == {}
    else:
        with pytest.raises(RuntimeError, match="lacks verified official"):
            retry.prepare(root, RUN, ["failed"], recovery(root))


def test_commit_error_rolls_back_all_artifacts(project, monkeypatch):
    root, _, _ = project
    before = active_snapshot(root)
    retry.prepare(root, RUN, ["failed"], recovery(root))
    original = retry.shutil.copyfile
    fired = False
    def fail_once(source, destination, *args, **kwargs):
        nonlocal fired
        if str(destination).endswith(".retry-apply") and not fired:
            fired = True
            raise OSError("injected commit failure")
        return original(source, destination, *args, **kwargs)
    monkeypatch.setattr(retry.shutil, "copyfile", fail_once)
    with pytest.raises(OSError, match="injected commit failure"):
        retry.apply(root, recovery(root))
    assert active_snapshot(root) == before
    assert retry.read(recovery(root) / "receipt.json")["status"] == "rolled_back"


def test_reject_archive_retry(project):
    root, layout, _ = project
    path = layout.path("raw_inventory.jsonl")
    data = [row for _, row in retry.rows(path)]
    data[1]["file_family"] = "archive"
    path.write_text("".join(json.dumps(row) + "\n" for row in data))
    with pytest.raises(RuntimeError, match="Archive retry"):
        retry.prepare(root, RUN, ["failed"], recovery(root))


def test_reject_duplicate_status(project):
    root, layout, _ = project
    path = layout.path("asset_parse_status.jsonl")
    first = next(retry.rows(path))[0]
    with path.open("ab") as stream:
        stream.write(first)
    with pytest.raises(RuntimeError, match="Duplicate asset status"):
        retry.prepare(root, RUN, ["failed"], recovery(root))


def test_reject_downstream_started(project):
    root, layout, _ = project
    path = layout.path("pipeline_state.json")
    state = retry.read(path)
    state["stages"]["deterministic_extract"]["status"] = "running"
    write(path, state)
    with pytest.raises(RuntimeError, match="Downstream processing"):
        retry.prepare(root, RUN, ["failed"], recovery(root))


def test_reject_unproved_metadata_only(project, monkeypatch):
    root, _, _ = project
    original = RawFormatParsers.parse_text

    def metadata_only(self, asset, emitter):
        if asset.raw_asset_id == "failed":
            return ParseOutcome(status="metadata_only", metadata={"llm_eligible": False})
        return original(self, asset, emitter)

    monkeypatch.setattr(RawFormatParsers, "parse_text", metadata_only)
    before = active_snapshot(root)
    with pytest.raises(RuntimeError):
        retry.prepare(root, RUN, ["failed"], recovery(root))
    assert active_snapshot(root) == before


@pytest.fixture
def publisher_metadata_project(project):
    root, layout, assets = project
    original = assets[1]
    path = original.physical_path.with_suffix(".xml")
    path.write_bytes(b"<data:data><datasets><author>fixture</author><type>0</type><dataset>"
                     b"<value>1</value><reason>Data will be made available on request</reason>"
                     b"<comments/></dataset></datasets></data:data>")
    data = path.read_bytes()
    metadata = {"byte_count": len(data), "status_code": 200,
        "metadata": {"s3_bucket": "pmc-oa-opendata", "s3_key": "PMC1234567.1/mmc1.xml",
                     "s3_size": len(data), "s3_etag": hashlib.md5(data).hexdigest()}}
    asset = replace(original, storage_path=str(path), source_id="pmc_oa", compound=None,
        source_url="https://pmc-oa-opendata.s3.amazonaws.com/PMC1234567.1/mmc1.xml",
        checksum=retry.digest(path), content_type="application/xml", file_extension=".xml",
        file_family="xml", file_size=len(data), asset_metadata_json=metadata)
    path_inventory = layout.path("raw_inventory.jsonl")
    inventory = [row for _, row in retry.rows(path_inventory)]
    inventory[1] = asset.__dict__
    path_inventory.write_text("".join(json.dumps(row) + "\n" for row in inventory))
    for name in retry.LEDGERS:
        ledger = layout.path(name)
        values = [row for _, row in retry.rows(ledger)]
        for row in values:
            if row["raw_asset_id"] == asset.raw_asset_id:
                row.update(checksum=asset.checksum, storage_path=asset.storage_path, source_id=asset.source_id,
                           compound=None, parser_name="pmc_jats_parser")
                if name == "asset_parse_status.jsonl":
                    row["file_family"] = "xml"
                else:
                    row["source_url"] = asset.source_url
        ledger.write_text("".join(json.dumps(row) + "\n" for row in values))
    return root, layout, asset


def test_publisher_metadata_retry_keeps_source_and_only_adds_metadata(publisher_metadata_project):
    root, layout, asset = publisher_metadata_project
    before = active_snapshot(root)
    source = asset.physical_path.read_bytes()
    receipt = retry.prepare(root, RUN, [asset.raw_asset_id], recovery(root))
    assert active_snapshot(root) == before
    assert asset.physical_path.read_bytes() == source
    assert receipt["summary"]["accepted"] is True
    assert receipt["summary"]["terminal_status_counts"]["metadata_only"] == 1
    assert list(receipt["append"]) == [str(layout.path("parsed_records.jsonl").relative_to(root))]
    retry.apply(root, recovery(root))
    assert asset.physical_path.read_bytes() == source
    for name in ParseEmitter.OUTPUT_NAMES:
        if name not in retry.LEDGERS and name != "parsed_records.jsonl":
            assert layout.path(name).read_bytes() == before[str(layout.path(name).relative_to(root))]
    metadata = [row for _, row in retry.rows(layout.path("parsed_records.jsonl"))
                if row["raw_asset_id"] == asset.raw_asset_id]
    assert len(metadata) == 1
    assert metadata[0]["record_type"] == "publisher_data_availability"
    assert metadata[0]["checksum"] == asset.checksum
    assert metadata[0]["source_url"] == asset.source_url
    retry.apply(root, recovery(root), rollback=True)
    assert active_snapshot(root) == before


@pytest.mark.parametrize("mutation", ["no_record", "extra_document", "record_type", "source_id", "checksum",
                                      "record_hash", "record_json", "llm_eligible", "source_location"])
def test_publisher_metadata_retry_rejects_nonmetadata_or_wrong_provenance(
        publisher_metadata_project, monkeypatch, mutation):
    root, _, asset = publisher_metadata_project
    original = RawFormatParsers.parse_pmc_xml

    def mutated(self, current_asset, emitter):
        write_original = emitter.write

        def write(name, row):
            if name == "parsed_records.jsonl" and row.get("raw_asset_id") == current_asset.raw_asset_id:
                if mutation == "no_record":
                    return
                row = dict(row)
                if mutation in {"record_type", "source_id", "checksum", "record_hash", "source_location"}:
                    row[mutation] = "changed"
                elif mutation == "record_json":
                    row[mutation] = {"availability_statement": "Invented scientific statement"}
            return write_original(name, row)

        emitter.write = write
        outcome = original(self, current_asset, emitter)
        if mutation == "llm_eligible":
            outcome.metadata["llm_eligible"] = True
        if mutation == "extra_document":
            emitter.document(current_asset, parser_name="pmc_jats_parser", content_kind="xml_text",
                title="Incorrect scientific input", text="This metadata must never become evidence or an extraction request.",
                source_location=emitter.location(current_asset, "xml_path=/"), metadata={"llm_eligible": True})
        emitter.write = write_original
        return outcome

    monkeypatch.setattr(RawFormatParsers, "parse_pmc_xml", mutated)
    before = active_snapshot(root)
    with pytest.raises(RuntimeError, match="Publisher metadata"):
        retry.prepare(root, RUN, [asset.raw_asset_id], recovery(root))
    assert active_snapshot(root) == before
