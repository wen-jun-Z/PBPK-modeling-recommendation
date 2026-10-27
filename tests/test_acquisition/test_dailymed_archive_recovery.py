import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest

from pbpk_kb.acquisition.dailymed_recovery import register_verified_archive_replacement, REASON

SETID = "0720268c-2c92-4632-8c55-76c254d1beb8"
DOCID = "2679593b-243b-42cd-9df8-09611c35bb3b"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def case(tmp_path):
    run = tmp_path / "raw_run"
    original_path = run / "compounds/sildenafil/bad.xml"
    original_path.parent.mkdir(parents=True)
    original_path.write_bytes(b"None")
    original = {
        "asset_id": "asset:original", "source_id": "dailymed", "compound": "sildenafil",
        "group": "dailymed/spl_xml", "status_code": 200,
        "url": f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{SETID}.xml",
        "request_id": f"dailymed:spl_xml_{SETID}:0000001", "request_key": f"spl_xml_{SETID}",
        "storage_path": str(original_path), "metadata_path": str(original_path) + ".metadata.json",
        "byte_count": 4, "checksum": sha256(b"None").hexdigest(),
        "metadata": {"setid": SETID, "spl_version": 2, "title": "SILDENAFIL"},
    }
    write_json(Path(original["metadata_path"]), original)
    (run / "assets.jsonl").write_text(json.dumps(original) + "\n")
    selected = {"compound": "sildenafil", "source_id": "dailymed", "source_record_id": SETID,
                "decision": "fulltext_selected", "record": original["metadata"]}
    write_json(run / "compounds/sildenafil/selection/dailymed/fulltext_selected_records.jsonl", selected)
    archive = tmp_path / "official.zip"
    report = {"url": f"https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?type=zip&setid={SETID}&version=2",
              "status_code": 200, "path": str(archive)}
    report["final_url"] = report["url"]
    report_path = tmp_path / "probe.json"

    def zip_contents(*, setid=SETID, version=2, extra=None):
        xml = (f'<document xmlns="urn:hl7-org:v3"><id root="{DOCID}"/>'
               f'<setId root="{setid}"/><versionNumber value="{version}"/></document>').encode()
        with ZipFile(archive, "w") as zipped:
            zipped.writestr(DOCID + ".xml", xml)
            if extra:
                zipped.writestr(*extra)
        report["bytes"] = archive.stat().st_size
        report["sha256"] = sha256(archive.read_bytes()).hexdigest()
        write_json(report_path, report)

    zip_contents()
    return tmp_path, run, original, report_path, report, zip_contents


def recover(case):
    root, run, original, report_path, _, _ = case
    return register_verified_archive_replacement(run, original["asset_id"], report_path, root=root)


def test_exact_archive_recovers_without_rewriting_source_and_is_idempotent(case):
    root, run, original, report_path, report, _ = case
    ledger_before = (run / "assets.jsonl").read_bytes()
    sidecar_before = Path(original["metadata_path"]).read_bytes()
    receipt = recover(case)
    assert receipt["status"] == "completed"
    assets = rows(run / "assets.jsonl")
    assert len(assets) == 2
    assert assets[0]["processing_policy"] == "metadata_only"
    assert assets[0]["processing_policy_reason"] == REASON
    assert assets[1]["url"] == report["url"]
    assert assets[1]["metadata"]["setid"] == SETID
    assert assets[1]["metadata"]["replaces_invalid_asset_id"] == original["asset_id"]
    assert Path(original["storage_path"]).read_bytes() == b"None"
    assert Path(original["metadata_path"]).read_bytes() == sidecar_before
    assert Path(receipt["original_ledger_backup"]).read_bytes() == ledger_before
    assert Path(receipt["original_sidecar_backup"]).read_bytes() == sidecar_before
    final_ledger = (run / "assets.jsonl").read_bytes()
    assert recover(case) == receipt
    assert (run / "assets.jsonl").read_bytes() == final_ledger


@pytest.mark.parametrize("change", ["setid", "version", "sha", "none", "source", "selected_version", "official_url", "http", "traversal", "duplicate_xml"])
def test_rejects_unverified_identity_or_payload_without_changing_ledger(case, change):
    _, run, original, report_path, report, make_zip = case
    if change == "setid":
        make_zip(setid=DOCID)
    elif change == "version":
        make_zip(version=3)
    elif change == "sha":
        report["sha256"] = "0" * 64
        write_json(report_path, report)
    elif change == "none":
        Path(original["storage_path"]).write_bytes(b"null")
        original["checksum"] = sha256(b"null").hexdigest()
        (run / "assets.jsonl").write_text(json.dumps(original) + "\n")
    elif change == "source":
        original["source_id"] = "other"
        (run / "assets.jsonl").write_text(json.dumps(original) + "\n")
    elif change == "selected_version":
        path = run / "compounds/sildenafil/selection/dailymed/fulltext_selected_records.jsonl"
        selected = json.loads(path.read_text())
        selected["record"]["spl_version"] = 3
        write_json(path, selected)
    elif change == "official_url":
        report["url"] = report["url"].replace("dailymed.nlm.nih.gov", "untrusted.example")
        write_json(report_path, report)
    elif change == "http":
        report["status_code"] = 500
        write_json(report_path, report)
    elif change == "traversal":
        make_zip(extra=("../escape.txt", "unsafe"))
    elif change == "duplicate_xml":
        make_zip(extra=("other.xml", '<document xmlns="urn:hl7-org:v3"/>'))
    before = (run / "assets.jsonl").read_bytes()
    with pytest.raises(ValueError):
        recover(case)
    assert (run / "assets.jsonl").read_bytes() == before


def test_multiple_recoveries_keep_separate_provenance_and_remain_idempotent(case):
    root, run, original, report_path, report, _ = case
    first = recover(case)
    other = json.loads(json.dumps(original))
    other["asset_id"] = "asset:second"
    other["storage_path"] = str(run / "compounds/sildenafil/second.xml")
    other["metadata_path"] = other["storage_path"] + ".metadata.json"
    Path(other["storage_path"]).write_bytes(b"None")
    write_json(Path(other["metadata_path"]), other)
    with (run / "assets.jsonl").open("a") as stream:
        stream.write(json.dumps(other) + "\n")
    second = register_verified_archive_replacement(run, other["asset_id"], report_path, root=root)
    assert second["original_ledger_backup"] != first["original_ledger_backup"]
    assert len(rows(run / "assets.jsonl")) == 4
    assert recover(case) == first


def test_tampered_replacement_is_not_accepted_on_repeat(case):
    receipt = recover(case)
    run = case[1]
    replacement = rows(run / "assets.jsonl")[1]
    Path(replacement["storage_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="destination changed"):
        recover(case)
