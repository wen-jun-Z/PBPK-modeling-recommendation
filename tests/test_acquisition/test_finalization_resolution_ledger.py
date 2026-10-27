"""Zero failures must produce an explicit empty ledger, never false resolutions."""
import json
from pathlib import Path

import pytest

from pbpk_kb.acquisition.finalize_raw_run import RawRunFinalizer, ensure_failure_resolution_ledger
from pbpk_kb.acquisition.reconcile_final_acquisition import reconcile


def finalizer(tmp_path):
    root = tmp_path / "compound_runs" / "run_test"
    root.mkdir(parents=True)
    for name in ("assets.jsonl", "failures.jsonl"):
        (root / name).write_bytes(b"")
    return RawRunFinalizer(run_root=root, output_root=tmp_path, sources_yaml=Path("unused.yaml"),
        compounds=["midazolam"], timeout_seconds=1, delay_seconds=0, pmc_download_workers=1,
        max_pmc_object_bytes=0, finish_shared_sources=False, shared_source_ids=[])


def test_zero_failure_finalization_creates_empty_ledger_before_completion(tmp_path):
    worker = finalizer(tmp_path)
    assert not worker.resolutions_path.exists()
    summary = worker.finalize()
    assert summary["failure_count"] == summary["failure_resolution_count"] == 0
    assert worker.resolutions_path.read_bytes() == b""
    assert json.loads(worker.status_path.read_text())["status"] == "completed"
    before = worker.resolutions_path.stat()
    worker.finalize()
    after = worker.resolutions_path.stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_existing_resolution_bytes_preserved_by_both_upstream_entrypoints(tmp_path):
    worker = finalizer(tmp_path)
    worker.resolutions_path.parent.mkdir()
    original = b'{ "original_failure_request_id" : "old", "status" : "resolved", "detail" : "prior verified recovery" }\n\n'
    worker.resolutions_path.write_bytes(original)
    worker.failures_path.write_text(json.dumps({"request_id": "old", "source_id": "pubchem"}) + "\n")
    before = worker.resolutions_path.stat()
    summary = worker.finalize()
    assert summary["failure_resolution_count"] == 1
    assert summary["remaining_unresolved_historical_failure_count"] == 0
    result = reconcile(worker.run_root, tmp_path, worker.compounds)
    assert result["failure_resolution_count"] == 1
    assert worker.resolutions_path.read_bytes() == original
    after = worker.resolutions_path.stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)


def test_empty_ledger_does_not_resolve_an_actual_failed_download(tmp_path):
    worker = finalizer(tmp_path)
    failure = {"request_id": "not-recovered", "source_id": "pubchem", "error": "download failed"}
    worker.failures_path.write_text(json.dumps(failure) + "\n")
    worker.finalize()
    result = reconcile(worker.run_root, tmp_path, worker.compounds)
    assert worker.resolutions_path.read_bytes() == b""
    assert result["accepted_as_complete_raw_content"] is False
    assert result["remaining_unresolved_failure_count"] == 1
    assert result["failure_resolution_count"] == 0


def test_directory_in_place_of_ledger_is_not_silently_accepted(tmp_path):
    path = tmp_path / "finalization" / "failure_resolutions.jsonl"
    path.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        ensure_failure_resolution_ledger(tmp_path)
