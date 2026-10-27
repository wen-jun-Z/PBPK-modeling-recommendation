"""Reversible, frozen-list withdrawal of 34 context reviews and 2 wrong analytes.

No source/final artifacts or scientific values are rewritten. Rehearsal always
rolls back. Apply requires a successful rehearsal of identical database rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import revise_expansion_evidence as r

AUDIT = r.RECOVERY / "current_serving_context_audit.json"
AUDIT_SHA = "8f1057015225de15bf42df85a2c0a7f2edbd73a8233754a692e25be9bb6e40cf"
ARCHIVES = r.RECOVERY / "serving_quarantine"
MANUAL = "fcde9cee-72a2-5c47-bc34-da896023230a"
NAMED_ANALYTE = "9b12033f-608e-50e6-82d2-3341ae45381b"
MUTABLE = {"processed_evidence_candidate", "clinical_pk_evidence", "evidence_layer_membership",
           "rag_retrieval_record", "enzyme_transporter_relationship"}


def frozen_decisions():
    if r.sha(AUDIT) != AUDIT_SHA:
        raise RuntimeError("Frozen source-context audit changed")
    audit = json.loads(AUDIT.read_text())
    selected = {v["evidence_id"]: v for v in audit["rows"] if v["domain_gate_status"] != "passed" or v["evidence_id"] == MANUAL}
    if (len(selected) != 36 or audit["row_count"] != 750
            or Counter(v["domain_gate_status"] for v in selected.values()) != {"needs_manual_review": 34, "excluded": 1, "passed": 1}
            or audit["confirmed_table_analyte_errors_from_independent_manual_source_check"] != [MANUAL]
            or selected[NAMED_ANALYTE]["domain_gate_reasons"] != ["clearance_value_belongs_to_other_named_analyte"]
            or any(v["run_id"] not in r.RUNS or v["evidence_type"] != "clinical_pk" for v in selected.values())):
        raise RuntimeError("Quarantine scope differs from reviewed 34 context + 2 analyte decisions")
    return selected


def serving_counts(c):
    return dict(c.execute("SELECT CASE WHEN run_id LIKE 'run_20261009%' THEN 'new70' ELSE 'old30' END,count(*) FROM auto_validated_evidence GROUP BY 1"))


def scope_and_sources(c, selected):
    if serving_counts(c) != {"old30": 2872, "new70": 750}:
        raise RuntimeError("Serving baseline changed; refuse stale quarantine")
    rows = {v["evidence_id"]: v for (v,) in c.execute("SELECT to_jsonb(t) FROM auto_validated_evidence t WHERE evidence_id::text=ANY(%s)", (sorted(selected),))}
    if set(rows) != set(selected):
        raise RuntimeError("A reviewed row is no longer currently serving")
    sources = {}
    for eid, v in rows.items():
        expected = selected[eid]
        for key in ("run_id", "evidence_type", "compound_name", "source_id", "storage_path", "source_location", "parameter"):
            if v.get(key) != expected.get(key):
                raise RuntimeError("Reviewed row identity changed: " + eid + ":" + key)
        if str(v.get("value")) != str(expected.get("value")) or v.get("unit") != expected.get("unit"):
            raise RuntimeError("Reviewed scientific value changed: " + eid)
        path = r.inside(Path(v["storage_path"]))
        actual = sources.setdefault(str(path), r.sha(path))
        if actual != v["checksum"] or hashlib.sha256(v["source_text"].encode()).hexdigest() != v["source_text_hash"]:
            raise RuntimeError("Reviewed source/quote checksum mismatch: " + eid)
    rag = list(c.execute("SELECT retrieval_record_id::text,run_id,evidence_ids FROM rag_retrieval_record WHERE evidence_ids ?| %s", (sorted(selected),)))
    if any(run not in r.RUNS for _, run, _ in rag):
        raise RuntimeError("A protected historical RAG row links a new-batch withdrawal")
    shared = {str(eid) for _, _, ids in rag for eid in ids} - set(selected)
    if shared and c.execute("SELECT 1 FROM processed_evidence_candidate WHERE evidence_id::text=ANY(%s) AND NOT(run_id=ANY(%s)) LIMIT 1", (sorted(shared), sorted(r.RUNS))).fetchone():
        raise RuntimeError("Withdrawal would remove retrieval linked to original evidence")
    scope = {"evidence_ids": sorted(selected), "rag_ids": sorted(x[0] for x in rag),
             "shared_rag_other_evidence_ids": sorted(shared), "sources": sources,
             "decisions": {eid: {"run_id": v["run_id"], "kind": "analyte_error" if eid in {MANUAL, NAMED_ANALYTE} else "context_review",
                 "reasons": ["table_value_belongs_to_active_metabolite"] if eid == MANUAL else v["domain_gate_reasons"]}
                 for eid, v in selected.items()}}
    return scope


def selection(table, scope):
    if table == "rag_retrieval_record":
        return "retrieval_record_id::text=ANY(%s)", [scope["rag_ids"]]
    if table == "enzyme_transporter_relationship":
        return "processed_evidence_id::text=ANY(%s)", [scope["evidence_ids"]]
    if table in MUTABLE:
        return "evidence_id::text=ANY(%s)", [scope["evidence_ids"]]
    return "FALSE", []


def snapshot(c, scope, *, protected=False):
    from psycopg import sql
    result = {}
    for table, keys in r.PK.items():
        where, args = selection(table, scope)
        if protected:
            where = "NOT COALESCE((" + where + "),FALSE)"
        query = sql.SQL("SELECT {} FROM {} t WHERE {} ORDER BY {}").format(
            sql.SQL("md5(to_jsonb(t)::text)" if protected else "to_jsonb(t)::text"),
            sql.Identifier(table), sql.SQL(where), sql.SQL(",").join(map(sql.Identifier, keys)))
        result[table] = [v[0] for v in c.execute(query, args)]
    return result


def bundle_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def apply_rows(c, scope, revision):
    for eid, decision in scope["decisions"].items():
        note = json.dumps({**decision, "revision_id": revision, "source_audit_sha256": AUDIT_SHA})
        c.execute("UPDATE processed_evidence_candidate SET curation_status='needs_review',updated_at=now(),metadata_json=metadata_json || jsonb_build_object('serving_withdrawal',%s::jsonb) WHERE evidence_id::text=%s", (note, eid))
        c.execute("UPDATE clinical_pk_evidence SET curation_status='needs_review',updated_at=now() WHERE evidence_id::text=%s", (eid,))
        c.execute("DELETE FROM evidence_layer_membership WHERE evidence_id::text=%s AND evidence_layer IN ('auto_validated','accepted','modeling')", (eid,))
        if decision["kind"] == "analyte_error":
            c.execute("DELETE FROM evidence_layer_membership WHERE evidence_id::text=%s AND evidence_layer='reviewable'", (eid,))
        layer = "excluded" if decision["kind"] == "analyte_error" else "reviewable"
        c.execute("INSERT INTO evidence_layer_membership(evidence_id,run_id,evidence_layer,review_blockers,metadata_json) VALUES(%s,%s,%s,%s::jsonb,%s::jsonb) ON CONFLICT(evidence_id,evidence_layer) DO UPDATE SET review_blockers=EXCLUDED.review_blockers,metadata_json=evidence_layer_membership.metadata_json || EXCLUDED.metadata_json", (eid, decision["run_id"], layer, json.dumps(decision["reasons"]), json.dumps({"serving_withdrawal": json.loads(note)})))
    c.execute("DELETE FROM rag_retrieval_record WHERE retrieval_record_id::text=ANY(%s)", (scope["rag_ids"],))
    c.execute("DELETE FROM enzyme_transporter_relationship WHERE processed_evidence_id::text=ANY(%s)", (scope["evidence_ids"],))


def verify(c, scope):
    if serving_counts(c) != {"old30": 2872, "new70": 714}:
        raise RuntimeError("Withdrawal serving counts differ from exact frozen partition")
    if c.execute("SELECT 1 FROM auto_validated_evidence WHERE evidence_id::text=ANY(%s)", (scope["evidence_ids"],)).fetchone():
        raise RuntimeError("Withdrawn evidence still serves")
    if c.execute("SELECT 1 FROM rag_retrieval_record WHERE evidence_ids ?| %s LIMIT 1", (scope["evidence_ids"],)).fetchone():
        raise RuntimeError("Withdrawn evidence is still retrieved")
    remaining = {str(eid) for (eid,) in c.execute("SELECT DISTINCT jsonb_array_elements_text(evidence_ids) FROM rag_retrieval_record WHERE run_id=ANY(%s)", (sorted(r.RUNS),))}
    return sorted(set(scope["shared_rag_other_evidence_ids"]) - remaining)


def restore_snapshot(c, before):
    from psycopg import sql
    scope = before["scope"]
    where, args = selection("evidence_layer_membership", scope)
    c.execute(sql.SQL("DELETE FROM evidence_layer_membership WHERE {}").format(sql.SQL(where)), args)
    for table in ("processed_evidence_candidate", "clinical_pk_evidence", "evidence_layer_membership", "rag_retrieval_record", "enzyme_transporter_relationship"):
        r.restore_rows(c, table, before["rows"][table])
    if snapshot(c, scope) != before["rows"]:
        raise RuntimeError("Restored quarantine rows differ from before-image")
    r.assert_protected(before["protected"], snapshot(c, scope, protected=True))
    if serving_counts(c) != {"old30": 2872, "new70": 750}:
        raise RuntimeError("Restored quarantine serving counts differ")


class RollbackOnly(Exception):
    pass


def run(action, rehearsal=None):
    import psycopg
    selected = frozen_decisions()
    with psycopg.connect(r.imp.psycopg_dsn(r.registered_url()), autocommit=True,
                         options="-c statement_timeout=120000 -c lock_timeout=10000") as c:
        try:
            with c.transaction():
                if action == "dry-run":
                    c.execute("SET TRANSACTION READ ONLY")
                else:
                    r.lock_project_tables(c)
                scope = scope_and_sources(c, selected)
                before = snapshot(c, scope)
                protected = snapshot(c, scope, protected=True)
                fingerprint = bundle_hash({"scope": scope, "before": before, "protected": protected})
                summary = {"action": action, "context_review_count": 34, "analyte_error_count": 2,
                    "withdrawn_count": 36, "rag_records_withdrawn": len(scope["rag_ids"]),
                    "shared_rag_other_evidence_count": len(scope["shared_rag_other_evidence_ids"]),
                    "baseline_sha256": fingerprint, "tool_sha256": r.sha(Path(__file__)), "audit_sha256": AUDIT_SHA}
                if action == "dry-run":
                    return summary
                if action == "apply":
                    receipt_path = r.inside(Path(rehearsal or ""))
                    if not receipt_path.is_relative_to(ARCHIVES):
                        raise RuntimeError("Apply requires a registered rehearsal receipt")
                    receipt = json.loads(receipt_path.read_text())
                    if any(receipt.get(k) != summary[k] for k in ("baseline_sha256", "tool_sha256", "audit_sha256")) or receipt.get("status") != "rolled_back_and_verified":
                        raise RuntimeError("Successful identical rehearsal required")
                revision = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
                archive = ARCHIVES / revision
                r.write_new(archive / "before.json", {"scope": scope, "rows": before, "protected": protected, "summary": summary})
                apply_rows(c, scope, revision)
                summary["other_valid_evidence_without_remaining_rag"] = verify(c, scope)
                r.assert_protected(protected, snapshot(c, scope, protected=True))
                r.write_new(archive / "after.json", {"scope": scope, "rows": snapshot(c, scope)})
                r.write_new(archive / "prepared.json", {"before_sha256": r.sha(archive / "before.json"), "after_sha256": r.sha(archive / "after.json"), "action": action})
                if action == "rehearse":
                    restore_snapshot(c, {"scope": scope, "rows": before, "protected": protected})
                    raise RollbackOnly()
        except RollbackOnly:
            with c.transaction():
                c.execute("SET TRANSACTION READ ONLY")
                if snapshot(c, scope) != before:
                    raise RuntimeError("Rollback did not restore exact target rows")
                r.assert_protected(protected, snapshot(c, scope, protected=True))
                if serving_counts(c) != {"old30": 2872, "new70": 750}:
                    raise RuntimeError("Rollback baseline not restored")
            summary.update(status="rolled_back_and_verified", protected_rows=sum(map(len, protected.values())))
            r.write_new(archive / "rehearsal.json", summary)
            return {**summary, "receipt": str(archive / "rehearsal.json")}
        with c.transaction():
            c.execute("SET TRANSACTION READ ONLY")
            verify(c, scope)
            r.assert_protected(protected, snapshot(c, scope, protected=True))
        summary.update(status="committed_and_verified", protected_rows=sum(map(len, protected.values())), serving_counts={"old30": 2872, "new70": 714})
        r.write_new(archive / "committed.json", summary)
        return {**summary, "audit_dir": str(archive)}


def rollback(archive):
    import psycopg
    archive = r.inside(Path(archive))
    if not archive.is_relative_to(ARCHIVES):
        raise RuntimeError("Unregistered quarantine archive")
    prepared = json.loads((archive / "prepared.json").read_text())
    if prepared["action"] != "apply" or any(r.sha(archive / (name + ".json")) != prepared[name + "_sha256"] for name in ("before", "after")):
        raise RuntimeError("Invalid quarantine before/after receipt")
    before, after = [json.loads((archive / (name + ".json")).read_text()) for name in ("before", "after")]
    scope = before["scope"]
    if scope != after["scope"] or set(scope["evidence_ids"]) != set(frozen_decisions()):
        raise RuntimeError("Rollback scope differs from frozen audit")
    with psycopg.connect(r.imp.psycopg_dsn(r.registered_url()), autocommit=True, options="-c statement_timeout=120000 -c lock_timeout=10000") as c:
        with c.transaction():
            r.lock_project_tables(c)
            if snapshot(c, scope) != after["rows"]:
                raise RuntimeError("Rows changed after quarantine; refuse stale rollback")
            r.assert_protected(before["protected"], snapshot(c, scope, protected=True))
            restore_snapshot(c, before)
        r.write_new(archive / "rolled_back.json", {"status": "restored_exactly"})
    return {"status": "restored_exactly", "audit_dir": str(archive)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("dry-run", "rehearse", "apply", "rollback"))
    parser.add_argument("--rehearsal")
    parser.add_argument("--audit-dir")
    args = parser.parse_args()
    result = rollback(args.audit_dir) if args.action == "rollback" else run(args.action, args.rehearsal)
    print(json.dumps(result, ensure_ascii=False))
