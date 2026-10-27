import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/deployment"))
import quarantine_expansion_evidence as q


def test_frozen_decisions_exact_and_distinguish_review_from_error():
    rows = q.frozen_decisions()
    assert len(rows) == 36
    assert sum(r["domain_gate_status"] == "needs_manual_review" for r in rows.values()) == 34
    assert q.MANUAL in rows and q.NAMED_ANALYTE in rows


def test_changed_audit_cannot_expand_withdrawal_scope(tmp_path, monkeypatch):
    path = tmp_path / "audit.json"
    path.write_text(q.AUDIT.read_text() + "\n")
    monkeypatch.setattr(q, "AUDIT", path)
    with pytest.raises(RuntimeError, match="audit changed"):
        q.frozen_decisions()


@pytest.mark.parametrize("mutation", ["old_run", "wrong_type", "missing_review", "extra_manual"])
def test_even_rehashed_audit_must_keep_reviewed_scientific_scope(tmp_path, monkeypatch, mutation):
    content = json.loads(q.AUDIT.read_text())
    row = next(r for r in content["rows"] if r["domain_gate_status"] == "needs_manual_review")
    if mutation == "old_run":
        row["run_id"] = "run_20260909T083120Z"
    elif mutation == "wrong_type":
        row["evidence_type"] = "public_pbpk_model"
    elif mutation == "missing_review":
        row["domain_gate_status"] = "passed"
    else:
        content["confirmed_table_analyte_errors_from_independent_manual_source_check"].append(row["evidence_id"])
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(content))
    monkeypatch.setattr(q, "AUDIT", path)
    monkeypatch.setattr(q, "AUDIT_SHA", q.r.sha(path))
    with pytest.raises(RuntimeError, match="scope differs"):
        q.frozen_decisions()


def test_quarantine_never_selects_models_registry_sources_or_duplicate_groups():
    scope = {"evidence_ids": [q.MANUAL], "rag_ids": ["rag-example"]}
    for table in set(q.r.PK) - q.MUTABLE:
        assert q.selection(table, scope) == ("FALSE", [])
    assert q.selection("rag_retrieval_record", scope)[1] == [["rag-example"]]
    assert q.selection("clinical_pk_evidence", scope)[1] == [[q.MANUAL]]


def test_stale_serving_baseline_is_rejected_before_source_work(monkeypatch):
    monkeypatch.setattr(q, "serving_counts", lambda c: {"old30": 2872, "new70": 714})
    with pytest.raises(RuntimeError, match="baseline changed"):
        q.scope_and_sources(object(), q.frozen_decisions())


def test_rollback_cannot_use_unregistered_path(tmp_path):
    with pytest.raises(RuntimeError, match="escapes project|Unregistered"):
        q.rollback(tmp_path)
