"""Revision planning and transaction tests; never connect to the project DB."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from pbpk_kb.storage.expansion_identity_mapping import ExpansionIdentityMapping

PATH = Path(__file__).resolve().parents[2] / "scripts/deployment/revise_expansion_evidence.py"
spec = importlib.util.spec_from_file_location("revision_tool", PATH)
revision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(revision)
RUN = "run_20261009T140919Z"
EID = "11111111-1111-4111-8111-111111111111"
CID = "22222222-2222-4222-8222-222222222222"
PID = "33333333-3333-4333-8333-333333333333"
RID = "44444444-4444-4444-8444-444444444444"


def json_file(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def jsonl_file(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def source_fixture(tmp_path):
    raw, final = tmp_path / "raw", tmp_path / "final"
    text = "Caffeine Cmax was 10 ng/mL in healthy adults."
    row = {"evidence_id": EID, "run_id": RUN, "compound": "caffeine", "compound_name": "caffeine",
           "compound_id": CID, "attributed_compound_id": CID, "attributed_compound_name": "caffeine",
           "compound_role": "target_compound", "source_id": "pubmed", "source_url": "https://pubmed.ncbi.nlm.nih.gov/1/",
           "raw_asset_id": "asset:1", "parsed_document_id": PID, "source_location": "pubmed:1#abstract",
           "storage_path": str(tmp_path / "raw.xml"), "checksum": "a" * 64, "source_text": text,
           "source_text_hash": hashlib.sha256(text.encode()).hexdigest(), "extraction_method": "rule_based",
           "extractor_version": "1", "extraction_confidence": 0.9, "parameter": "Cmax", "value": 10,
           "unit": "ng/mL", "canonical_value": 10, "canonical_unit": "ng/mL", "evidence_type": "clinical_pk",
           "quality_status": "auto_validated", "curation_status": "auto_validated", "domain_gate_status": "passed"}
    layout = revision.imp.raw_processing_layout(raw)
    jsonl_file(layout.path("evidence_candidates.jsonl", for_write=True), [row])
    jsonl_file(layout.path("raw_inventory.jsonl", for_write=True), [{"raw_asset_id": "asset:1", "source_id": "pubmed",
        "storage_path": row["storage_path"], "checksum": row["checksum"], "source_url": row["source_url"]}])
    jsonl_file(layout.path("parsed_documents.jsonl", for_write=True), [{"parsed_document_id": PID, "raw_asset_id": "asset:1",
        "source_id": "pubmed", "source_location": row["source_location"], "text_hash": row["source_text_hash"], "run_id": RUN}])
    json_file(final / "00_run_manifest/pipeline_state.json", {"run_id": RUN})
    jsonl_file(final / "01_compound_attribution/compound_identity.jsonl", [{"compound_id": CID, "preferred_name": "caffeine"}])
    jsonl_file(final / "01_compound_attribution/compound_attribution.jsonl", [{"evidence_id": EID, "compound_id": CID,
        "compound_name": "caffeine", "compound_role": "target_compound", "attribution_method": "rule"}])
    for name in ["03_deduplication/deterministic_duplicate_groups.jsonl", "03_deduplication/deterministic_duplicate_members.jsonl",
                 "03_deduplication/semantic_duplicate_groups.jsonl", "03_deduplication/semantic_duplicate_members.jsonl",
                 "04_conflict_analysis/evidence_conflict_report.jsonl", "04_conflict_analysis/conflict_members.jsonl",
                 "05_evidence_layers/excluded_from_reviewable.jsonl", "07_llm_quality_validation/final_quarantined_evidence.jsonl"]:
        jsonl_file(final / name, [])
    jsonl_file(final / "05_evidence_layers/reviewable_evidence.jsonl", [row])
    jsonl_file(final / "07_llm_quality_validation/final_auto_validated_evidence.jsonl", [row])
    json_file(final / "07_llm_quality_validation/quality_summary.json", {"accepted": True, "final_auto_validated_count": 1})
    index = final / "06_rag_index"
    jsonl_file(index / "retrieval_records.jsonl", [{"retrieval_record_id": RID, "run_id": RUN, "embedding_row": 0,
        "compound_id": CID, "compound_name": "caffeine", "evidence_ids": [EID], "source_id": "pubmed",
        "source_location": row["source_location"], "source_text": text, "source_text_hash": row["source_text_hash"],
        "embedding_text": text, "record_kinds": ["evidence"], "evidence_layer": "auto_validated"}])
    np.save(index / "retrieval_embeddings.npy", np.ones((1, 768), dtype=np.float32) / np.sqrt(768))
    json_file(index / "rag_index_summary.json", {"accepted": True, "embedding_dimension": 768, "record_count": 1,
        "embedding_model": "local_fixture", "records_sha256": revision.sha(index / "retrieval_records.jsonl"),
        "embeddings_sha256": revision.sha(index / "retrieval_embeddings.npy")})
    return raw, final, row


@pytest.mark.parametrize("bad_run", ["run_20260713T163840Z", "run_20260909T083120Z", "run_20261009T140919Z-extra", None])
def test_only_frozen_new_runs_are_eligible(bad_run):
    with pytest.raises(RuntimeError, match="seven frozen"):
        revision.check_run(bad_run)


def test_collection_reuses_transformations_without_any_database_writer(tmp_path, monkeypatch):
    raw, final, row = source_fixture(tmp_path)
    def forbidden(*a, **kw):
        raise AssertionError("Database writer called while collecting")
    monkeypatch.setattr(revision.imp, "CopyWriter", forbidden)
    monkeypatch.setattr(revision.imp, "create_db_engine", forbidden)
    collected = revision.RevisionRows(RUN, raw, final, ExpansionIdentityMapping()).collect()
    assert collected.serving_ids == {EID}
    assert float(collected.tables["processed_evidence_candidate"]["rows"][0]["value"]) == 10
    assert collected.tables["clinical_pk_evidence"]["rows"][0]["evidence_id"] == EID
    assert len(collected.tables["rag_retrieval_record"]["rows"]) == 1
    assert "compound_identity" not in collected.tables
    assert collected.native_database_url is None


@pytest.mark.parametrize("failure", ["quarantine_overlap", "wrong_run", "wrong_compound", "review", "rag_checksum"])
def test_bad_reprocessed_partition_is_rejected_before_writes(tmp_path, failure):
    raw, final, row = source_fixture(tmp_path)
    if failure == "quarantine_overlap":
        jsonl_file(final / "07_llm_quality_validation/final_quarantined_evidence.jsonl", [row])
    elif failure == "review":
        row["domain_gate_status"] = "needs_manual_review"
        jsonl_file(final / "07_llm_quality_validation/final_auto_validated_evidence.jsonl", [row])
    elif failure == "wrong_run":
        row["run_id"] = "run_20260909T083120Z"
        jsonl_file(revision.imp.raw_processing_layout(raw).path("evidence_candidates.jsonl"), [row])
    elif failure == "wrong_compound":
        jsonl_file(final / "01_compound_attribution/compound_attribution.jsonl", [{"evidence_id": EID, "compound_id": RID,
            "compound_name": "other drug", "compound_role": "target_compound"}])
    else:
        (final / "06_rag_index/retrieval_records.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError):
        revision.RevisionRows(RUN, raw, final, ExpansionIdentityMapping()).collect()


class LocalTransactionConnection:
    """Use real SQLite rollback/commit to test the tool's transaction boundary."""
    def __init__(self):
        self.db = sqlite3.connect(":memory:", isolation_level=None)
        self.db.execute("CREATE TABLE probe(id text PRIMARY KEY, value text)")
        self.db.executemany("INSERT INTO probe VALUES(?,?)", [("old", "untouched"), ("new", "before")])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN")
        try:
            yield self
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        else:
            self.db.execute("COMMIT")


@pytest.mark.parametrize("failure", [None, "serving_check", "protected_row"])
def test_apply_commits_all_or_rolls_back_on_final_validation(tmp_path, monkeypatch, failure):
    import psycopg
    connection = LocalTransactionConnection()
    monkeypatch.setattr(revision, "ROOT", tmp_path)
    monkeypatch.setattr(revision, "RECOVERY", tmp_path / "recovery")
    monkeypatch.setattr(revision, "registered_url", lambda: "postgresql://fixture-never-connected")
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: connection)
    rows = SimpleNamespace(run_dir=tmp_path / "raw", final_evidence_dir=tmp_path / "final", serving_ids={EID},
        tables={}, import_identity_mapping=ExpansionIdentityMapping())
    monkeypatch.setattr(revision, "collect", lambda *a: (rows, {}))
    monkeypatch.setattr(revision, "input_fingerprints", lambda *a: {})
    monkeypatch.setattr(revision, "lock_project_tables", lambda *a: None)
    monkeypatch.setattr(revision, "scope_for", lambda *a: {"run_id": RUN, "old_ids": [EID], "serving_ids": [EID]})
    def snap(conn, scope, *, hashes=False, protected=False):
        key = "old" if protected else "new"
        return {"probe": [conn.db.execute("SELECT value FROM probe WHERE id=?", (key,)).fetchone()[0]]}
    monkeypatch.setattr(revision, "snapshot", snap)
    def mutate(conn, *args):
        conn.db.execute("UPDATE probe SET value='after' WHERE id='new'")
        if failure == "protected_row":
            conn.db.execute("UPDATE probe SET value='wrong' WHERE id='old'")
    monkeypatch.setattr(revision, "apply_rows", mutate)
    def verify(*a):
        if failure == "serving_check":
            raise RuntimeError("serving mismatch")
    monkeypatch.setattr(revision, "verify_serving", verify)
    if failure:
        with pytest.raises(RuntimeError):
            revision.revise(RUN, apply=True)
        assert connection.db.execute("SELECT value FROM probe WHERE id='new'").fetchone()[0] == "before"
        assert not list(tmp_path.rglob("committed.json"))
    else:
        result = revision.revise(RUN, apply=True)
        assert result["protected_rows_unchanged"]
        assert connection.db.execute("SELECT value FROM probe WHERE id='new'").fetchone()[0] == "after"
        assert len(list(tmp_path.rglob("committed.json"))) == 1
    assert connection.db.execute("SELECT value FROM probe WHERE id='old'").fetchone()[0] == "untouched"


def test_snapshot_archive_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(revision, "ROOT", tmp_path)
    path = tmp_path / "before.json"
    revision.write_new(path, {"original": True})
    with pytest.raises(FileExistsError):
        revision.write_new(path, {"original": False})
    assert json.loads(path.read_text()) == {"original": True}


def test_dry_run_does_not_lock_write_or_archive(tmp_path, monkeypatch):
    import psycopg
    connection = LocalTransactionConnection()
    monkeypatch.setattr(revision, "registered_url", lambda: "postgresql://fixture-never-connected")
    options = []
    def connect(*a, **kw):
        options.append(kw["options"])
        return connection
    monkeypatch.setattr(psycopg, "connect", connect)
    rows = SimpleNamespace(serving_ids={EID}, tables={}, import_identity_mapping=ExpansionIdentityMapping())
    monkeypatch.setattr(revision, "collect", lambda *a: (rows, {}))
    monkeypatch.setattr(revision, "scope_for", lambda *a: {"old_ids": [], "serving_ids": [EID]})
    monkeypatch.setattr(revision, "snapshot", lambda *a, **kw: {})
    def forbidden(*a, **kw):
        raise AssertionError("Write attempted during dry-run")
    for name in ["lock_project_tables", "apply_rows", "write_new"]:
        monkeypatch.setattr(revision, name, forbidden)
    result = revision.revise(RUN)
    assert result["database_changed"] is False
    assert "default_transaction_read_only=on" in options[0]


def test_restore_keeps_original_decimal_lexeme():
    captured = []
    connection = SimpleNamespace(execute=lambda query, params: captured.append(params[0]))
    record = '{"evidence_id":"' + EID + '","value":1.2345678901234567890123456789}'
    revision.restore_rows(connection, "processed_evidence_candidate", [record])
    assert captured == ["[" + record + "]"]


def test_old_run_cannot_enter_through_rollback_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(revision, "ROOT", tmp_path)
    monkeypatch.setattr(revision, "RECOVERY", tmp_path / "recovery")
    audit = tmp_path / "recovery/db_revisions/test"
    before = {"scope": {"run_id": "run_20260909T083120Z"}}
    before_sha = revision.write_new(audit / "before.json", before)
    after_sha = revision.write_new(audit / "after.json", before)
    revision.write_new(audit / "transaction.json", {"before_sha256": before_sha, "after_sha256": after_sha})
    with pytest.raises(RuntimeError, match="seven frozen"):
        revision.rollback(audit)
