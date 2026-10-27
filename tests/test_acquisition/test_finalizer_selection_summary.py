"""Recovery counts come from durable selections, never a paused writer's memory."""
import json
from pathlib import Path

import pytest

from pbpk_kb.acquisition.finalize_raw_run import RawRunFinalizer, selection_summary_from_ledgers


def _selection(root, rows, source="openfda_drug_label", compound="sildenafil"):
    path = root / "compounds" / compound / "selection" / source / "candidate_records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _row(source="openfda_drug_label", **changes):
    return {"compound": "sildenafil", "source_id": source, "source_record_id": "label-1",
            "decision": "fulltext_selected", "record": {"id": "label-1"},
            "selected_at": "first", "request_id": "request-1", "source_url": "https://example.org/page1",
            **changes}


@pytest.mark.parametrize("source", ["pubmed", "dailymed", "clinicaltrials", "openfda_drug_label", "pkdb"])
def test_rebuilds_all_sources_and_deduplicates_transport_replays(tmp_path, source):
    original = _row(source)
    replay = {**original, "request_id": "retry-2", "selected_at": "later", "source_url": "https://example.org/retry"}
    _selection(tmp_path, [original, replay, _row(source, source_record_id="label-2", decision="metadata_only")], source)
    actual = selection_summary_from_ledgers(tmp_path, ["sildenafil", "vancomycin"])
    prefix = "openfda" if source == "openfda_drug_label" else source
    assert actual[f"{prefix}_selection_by_compound"] == {
        "sildenafil": {"fulltext_selected": 1, "metadata_only": 1}, "vancomycin": {}}


@pytest.mark.parametrize("changes", [{"decision": "metadata_only"}, {"record": {"id": "different-content"}}])
def test_conflicting_same_record_is_not_silently_deduplicated(tmp_path, changes):
    _selection(tmp_path, [_row(), _row(**changes)])
    with pytest.raises(ValueError, match="Conflicting selection records"):
        selection_summary_from_ledgers(tmp_path, ["sildenafil"])


@pytest.mark.parametrize("changes", [{"compound": "another-drug"}, {"source_id": "another-source"},
                                     {"source_record_id": ""}, {"decision": None}])
def test_invalid_selection_identity_or_decision_fails(tmp_path, changes):
    _selection(tmp_path, [_row(**changes)])
    with pytest.raises(ValueError, match="identity|decision"):
        selection_summary_from_ledgers(tmp_path, ["sildenafil"])


def test_malformed_ledger_is_not_skipped(tmp_path):
    path = _selection(tmp_path, [_row()])
    with path.open("a") as stream:
        stream.write('{"truncated":')
    with pytest.raises(ValueError, match="Malformed selection ledger"):
        selection_summary_from_ledgers(tmp_path, ["sildenafil"])


def test_selection_symlink_cannot_leave_run(tmp_path):
    root = tmp_path / "run"
    path = _selection(root, [_row()])
    outside = tmp_path / "outside.jsonl"
    outside.write_text(json.dumps(_row()) + "\n")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        selection_summary_from_ledgers(root, ["sildenafil"])


def test_finalizer_summary_includes_out_of_process_recovery(tmp_path):
    root = tmp_path / "compound_runs" / "run_test"
    _selection(root, [_row(), _row(source_record_id="label-2", decision="metadata_only")])
    # The paused writer could still have zero in-memory counts. Only these
    # persisted recovery selections are passed to the new finalizer instance.
    finalizer = RawRunFinalizer(run_root=root, output_root=tmp_path, sources_yaml=Path("unused.yaml"),
        compounds=["sildenafil"], timeout_seconds=30, delay_seconds=0, pmc_download_workers=1,
        max_pmc_object_bytes=0, finish_shared_sources=False, shared_source_ids=[])
    summary = finalizer.write_summary()
    assert summary["openfda_selection_by_compound"] == {
        "sildenafil": {"fulltext_selected": 1, "metadata_only": 1}}
    assert json.loads((root / "summary.json").read_text())["openfda_selection_by_compound"] == summary["openfda_selection_by_compound"]
