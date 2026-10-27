"""Rehearse, apply and verify a source-reviewed serving-only KB revision.

No original files, model catalog or experiment outputs are changed. Duplicate
and unresolved records remain in the database with before-images and reasons.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

import curate_legacy_evidence as curate
import revise_expansion_evidence as r
from verify_expansion_batch import _registered_database, _local_query_vectors

ORPHANS = {"5ce7c8e3-78ff-5c95-836d-662f75b29958", "61e3ca15-3273-5c07-bd6d-8bbb660261a0", "7abe1255-d105-55c2-8e5c-2dc107fa5945"}
MUTABLE = {"processed_evidence_candidate", "evidence_layer_membership", "rag_retrieval_record", "raw_document", "enzyme_transporter_relationship", *r.TYPED} - {"public_pbpk_model_registry"}
REVISION = "source_curation_20261022"


def hash_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def where(table, scope):
    if table == "rag_retrieval_record":
        return "retrieval_record_id::text=ANY(%s)", [scope["rag_ids"]]
    if table == "raw_document":
        return "raw_document_id::text=ANY(%s)", [scope["raw_ids"]]
    if table == "enzyme_transporter_relationship":
        return "processed_evidence_id::text=ANY(%s)", [scope["evidence_ids"]]
    if table in MUTABLE:
        return "evidence_id::text=ANY(%s)", [scope["evidence_ids"]]
    return "FALSE", []


def snapshot(c, scope, protected=False):
    from psycopg import sql
    result = {}
    for table, keys in r.PK.items():
        clause, args = where(table, scope)
        if protected:
            clause = "NOT COALESCE((" + clause + "),FALSE)"
        query = sql.SQL("SELECT {} FROM {} t WHERE {} ORDER BY {}").format(
            sql.SQL("md5(to_jsonb(t)::text)" if protected else "to_jsonb(t)::text"),
            sql.Identifier(table), sql.SQL(clause), sql.SQL(",").join(map(sql.Identifier, keys)))
        result[table] = [x[0] for x in c.execute(query, args)]
    return result


def update_fields(c, table, key, value, payload):
    from psycopg import sql
    from psycopg.types.json import Jsonb
    columns = list(payload)
    params = [Jsonb(payload[k]) if isinstance(payload[k], (dict, list)) else payload[k] for k in columns]
    c.execute(sql.SQL("UPDATE {} SET {} WHERE {}::text=%s").format(sql.Identifier(table),
        sql.SQL(",").join(sql.SQL("{}=%s").format(sql.Identifier(k)) for k in columns), sql.Identifier(key)), params + [value])


def prepare():
    import psycopg
    from pbpk_kb.processing.rag_index_stage import evidence_record, build_embedding_text
    proposal = curate.read(curate.BASE / "curation_proposal.json")
    items = proposal["items"]
    source_input = {x["evidence_id"]: x for x in curate.read(curate.BASE / "serving_input.json")}
    ids = sorted(x["evidence_id"] for x in items)
    active = {x["evidence_id"]: x["row"] for x in items if x["status"] == "accepted"}
    url, config = _registered_database()
    with psycopg.connect(r.imp.psycopg_dsn(url.render_as_string(hide_password=False)), options="-c default_transaction_read_only=on -c statement_timeout=60000") as c:
        current = {x["evidence_id"]: x for (x,) in c.execute("SELECT to_jsonb(t) FROM auto_validated_evidence t WHERE evidence_id::text=ANY(%s)", (ids,))}
        if any(hash_object(current.get(eid)) != hash_object(source_input[eid]) for eid in ids):
            raise RuntimeError("Serving rows changed since source audit")
        rag = [(rid, links) for rid, links in c.execute("SELECT retrieval_record_id::text,evidence_ids FROM rag_retrieval_record WHERE evidence_ids ?| %s", (ids,))]
        extra = ORPHANS | ({eid for _, links in rag for eid in links} - set(ids))
        for (row,) in c.execute("SELECT to_jsonb(t) FROM auto_validated_evidence t WHERE evidence_id::text=ANY(%s)", (sorted(extra),)):
            active[row["evidence_id"]] = row
        indices = dict(c.execute("SELECT run_id,max(embedding_row)+1 FROM rag_retrieval_record GROUP BY run_id"))
        c.rollback()
    new_raw = {}
    for item in items:
        row = item["row"]
        original = source_input[item["evidence_id"]]
        if row["storage_path"] != original["storage_path"] and item["source_recovery"] and not any(b.startswith("source_") or b == "original_source_not_recovered" for b in item["blockers"]):
            raw_id = str(uuid5(NAMESPACE_URL, REVISION + "|" + row["source_id"] + "|" + row["checksum"]))
            path = r.inside(Path(row["storage_path"]))
            if r.sha(path) != row["checksum"]:
                raise RuntimeError("Reviewed source bytes changed")
            new_raw[raw_id] = {"raw_document_id": raw_id, "source_id": row["source_id"], "source_native_id": "source-recovery:" + row["checksum"],
                "document_type": "source_recovery", "source_url": row.get("source_url"), "source_version": REVISION,
                "raw_storage_path": str(path), "raw_checksum": row["checksum"], "raw_size_bytes": path.stat().st_size,
                "metadata": {"method": item["source_recovery"].get("method", "registered_public_source_retrieval"),
                             "original_raw_document_id": original["raw_document_id"], "original_checksum": original["checksum"],
                             "byte_identity": row["checksum"] == original["checksum"]}}
            row["raw_document_id"] = raw_id
            row["raw_asset_id"] = "source-recovery:" + row["checksum"]
            row["parsed_document_id"] = None
            row["parsed_table_id"] = None
            row["chunk_id"] = None
    records = []
    for eid, row in sorted(active.items()):
        record = evidence_record(row, "auto_validated")
        record["embedding_text"] = build_embedding_text(record)
        record["retrieval_record_id"] = str(uuid5(NAMESPACE_URL, REVISION + "|" + eid))
        record["embedding_row"] = indices.get(row["run_id"], 0)
        indices[row["run_id"]] = record["embedding_row"] + 1
        records.append(record)
    vectors, revision = _local_query_vectors([x["embedding_text"] for x in records], config)
    for record in records:
        record["embedding"] = vectors[record["embedding_text"]]
    scope = {"evidence_ids": ids, "rag_ids": sorted({rid for rid, _ in rag} | {x["retrieval_record_id"] for x in records}),
             "raw_ids": sorted(new_raw), "new_raw_ids": sorted(new_raw), "new_rag_ids": [x["retrieval_record_id"] for x in records],
             "active_ids": sorted(active), "extra_ids": sorted(extra)}
    bundle = {"proposal_sha256": r.sha(curate.BASE / "curation_proposal.json"), "items": items, "scope": scope,
              "raw_documents": list(new_raw.values()), "rag_records": records, "embedding_revision": revision,
              "summary": proposal["summary"], "timestamp": datetime.now(timezone.utc).isoformat()}
    r.write_new(curate.BASE / "apply_bundle.json", bundle)
    print(json.dumps({"prepared": True, "active": len(active), "raw_sources": len(new_raw), "vectors": len(records)}))


def mutate(c, bundle):
    from psycopg.types.json import Jsonb
    from pbpk_kb.storage.import_processed_evidence import EvidenceImporter
    stamp = bundle["timestamp"]
    for raw in bundle["raw_documents"]:
        c.execute("INSERT INTO raw_document(raw_document_id,source_id,source_record_id,source_url,source_version,raw_storage_path,raw_checksum,raw_size_bytes,metadata,created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            [raw[k] for k in ("raw_document_id", "source_id", "source_native_id", "source_url", "source_version", "raw_storage_path", "raw_checksum", "raw_size_bytes")] + [Jsonb(raw["metadata"]), stamp])
    for item in bundle["items"]:
        eid, status, row = item["evidence_id"], item["status"], dict(item["row"])
        note = {"revision": REVISION, "status": status, "blockers": item["blockers"], "canonical_evidence_id": item.get("canonical_evidence_id"),
                "resolved_false_positives": item["resolved_false_positives"], "correction": item["correction"], "proposal_sha256": bundle["proposal_sha256"]}
        row["curation_status"] = "auto_validated" if status == "accepted" else "duplicate" if status == "duplicate" else "needs_review"
        row["metadata_json"] = {**row.get("metadata_json", {}), "source_curation": note}
        row["quality_status"] = row["curation_status"]
        row["fatal_blockers"] = item["blockers"]
        # Prior generated verdicts remain in before.json, not as current
        # verification of subsequently changed values.
        for stale in ("llm_validation", "llm_repair", "attribution", "layer_metadata_json"):
            row.pop(stale, None)
        payload = EvidenceImporter._candidate_row(None, row)
        payload.pop("evidence_id")
        for k in ("metadata_json", "source_record"):
            payload[k] = json.loads(payload[k])
        payload["updated_at"] = stamp
        update_fields(c, "processed_evidence_candidate", "evidence_id", eid, payload)
        if row["evidence_type"] != "public_pbpk_model":
            typed = EvidenceImporter._typed_row(None, row)
            if typed:
                table, values = typed
                values.pop("evidence_id")
                values["updated_at"] = stamp
                update_fields(c, table, "evidence_id", eid, values)
        if status != "accepted":
            c.execute("DELETE FROM evidence_layer_membership WHERE evidence_id::text=%s AND evidence_layer IN ('auto_validated','accepted','modeling')", (eid,))
            layer = "excluded" if status == "duplicate" else "reviewable"
            c.execute("INSERT INTO evidence_layer_membership(evidence_id,run_id,evidence_layer,review_blockers,metadata_json) VALUES (%s,%s,%s,%s,%s) ON CONFLICT(evidence_id,evidence_layer) DO UPDATE SET review_blockers=EXCLUDED.review_blockers,metadata_json=evidence_layer_membership.metadata_json || EXCLUDED.metadata_json",
                      (eid, row["run_id"], layer, Jsonb(item["blockers"]), Jsonb({"source_curation": note})))
            c.execute("DELETE FROM enzyme_transporter_relationship WHERE processed_evidence_id::text=%s", (eid,))
        else:
            c.execute("UPDATE evidence_layer_membership SET metadata_json=metadata_json || %s WHERE evidence_id::text=%s", (Jsonb({"source_curation": note}), eid))
    c.execute("DELETE FROM rag_retrieval_record WHERE retrieval_record_id::text=ANY(%s)", (bundle["scope"]["rag_ids"],))
    columns = r.imp.RAG_COLUMNS
    rag_rows = []
    for record in bundle["rag_records"]:
        payload = {k: record.get(k) for k in columns}
        payload.update(embedding=r.imp.vector_text(record["embedding"]), embedding_model="NeuML/pubmedbert-base-embeddings",
                       evidence_layer="auto_validated", curation_status="auto_validated", provenance_status="validated",
                       metadata_json=json.dumps({"source_curation": REVISION, "source_review_proposal_sha256": bundle["proposal_sha256"],
                                                 "parameters": record["parameters"], "values": record["values"]}))
        for key in ("record_kinds", "evidence_ids", "chunk_ids", "parsed_table_ids", "recovered_table_ids"):
            payload[key] = json.dumps(payload.get(key) or [])
        rag_rows.append(payload)
    r.copy_rows(c, "rag_retrieval_record", columns, rag_rows)


def verify(c, bundle):
    scope = bundle["scope"]
    expected = {x["evidence_id"] for x in bundle["items"] if x["status"] == "accepted"}
    active = {str(x[0]) for x in c.execute("SELECT evidence_id FROM auto_validated_evidence WHERE evidence_id::text=ANY(%s)", (scope["evidence_ids"],))}
    if active != expected:
        raise RuntimeError("Accepted serving partition differs")
    linked = {x[0] for x in c.execute("SELECT jsonb_array_elements_text(evidence_ids) FROM rag_retrieval_record WHERE retrieval_record_id::text=ANY(%s)", (scope["new_rag_ids"],))}
    if linked != set(scope["active_ids"]):
        raise RuntimeError("Search links differ from accepted partition")
    withdrawn = sorted(set(scope["evidence_ids"]) - expected)
    if c.execute("SELECT 1 FROM rag_retrieval_record WHERE evidence_ids ?| %s LIMIT 1", (withdrawn,)).fetchone():
        raise RuntimeError("Withdrawn or duplicate evidence remains searchable")
    return dict(c.execute("SELECT CASE WHEN run_id LIKE 'run_20261009%' THEN 'new70' ELSE 'old30' END,count(*) FROM auto_validated_evidence GROUP BY 1"))


def apply():
    import psycopg
    bundle = curate.read(curate.BASE / "apply_bundle.json")
    if r.sha(curate.BASE / "curation_proposal.json") != bundle["proposal_sha256"]:
        raise RuntimeError("Proposal changed after vector preparation")
    archive = curate.BASE / "transaction"
    if (archive / "committed.json").exists():
        raise RuntimeError("Revision is already committed")
    with psycopg.connect(r.imp.psycopg_dsn(r.registered_url()), autocommit=True,
                        options="-c statement_timeout=180000 -c lock_timeout=10000") as c:
        class RehearsalRollback(Exception):
            pass
        try:
            with c.transaction():
                r.lock_project_tables(c)
                before, protected = snapshot(c, bundle["scope"]), snapshot(c, bundle["scope"], True)
                original = {x["evidence_id"]: x for x in curate.read(curate.BASE / "serving_input.json")}
                current = {x["evidence_id"]: x for (x,) in c.execute("SELECT to_jsonb(t) FROM auto_validated_evidence t WHERE evidence_id::text=ANY(%s)", (bundle["scope"]["evidence_ids"],))}
                if any(hash_object(current.get(eid)) != hash_object(original[eid]) for eid in bundle["scope"]["evidence_ids"]):
                    raise RuntimeError("Stale curation snapshot")
                before_image = {"rows": before, "protected": protected, "scope": bundle["scope"], "bundle_sha256": r.sha(curate.BASE / "apply_bundle.json")}
                if (archive / "before.json").exists():
                    if curate.read(archive / "before.json") != before_image:
                        raise RuntimeError("Prior failed rehearsal no longer has the same baseline")
                else:
                    r.write_new(archive / "before.json", before_image)
                mutate(c, bundle)
                verify(c, bundle)
                r.assert_protected(protected, snapshot(c, bundle["scope"], True))
                raise RehearsalRollback()
        except RehearsalRollback:
            if snapshot(c, bundle["scope"]) != before:
                raise RuntimeError("Rehearsal rollback failed")
            r.assert_protected(protected, snapshot(c, bundle["scope"], True))
            r.write_new(archive / "rehearsal.json", {"status": "rolled_back_and_verified", "before_sha256": r.sha(archive / "before.json")})
        with c.transaction():
            r.lock_project_tables(c)
            if snapshot(c, bundle["scope"]) != before:
                raise RuntimeError("Baseline changed after rehearsal")
            r.assert_protected(protected, snapshot(c, bundle["scope"], True))
            mutate(c, bundle)
            counts = verify(c, bundle)
            r.assert_protected(protected, snapshot(c, bundle["scope"], True))
            r.write_new(archive / "after.json", {"rows": snapshot(c, bundle["scope"])})
        counts = verify(c, bundle)
        receipt = {"status": "committed_and_verified", "counts": counts, "review": bundle["summary"],
                   "retrieval_records": len(bundle["rag_records"]), "raw_documents": len(bundle["raw_documents"]),
                   "protected_rows_verified": sum(map(len, protected.values())), "rollback_rehearsed": True,
                   "before_sha256": r.sha(archive / "before.json"), "after_sha256": r.sha(archive / "after.json")}
        r.write_new(archive / "committed.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))


def repair_source_bindings():
    """Keep the candidate asset ID identical to its recovered raw registry ID."""
    import psycopg
    bundle = curate.read(curate.BASE / "apply_bundle.json")
    scope = bundle["scope"]
    archive = curate.BASE / "transaction"
    with psycopg.connect(r.imp.psycopg_dsn(r.registered_url()), autocommit=True) as c:
        with c.transaction():
            r.lock_project_tables(c)
            protected = snapshot(c, scope, True)
            before = snapshot(c, scope)
            if before != curate.read(archive / "after.json")["rows"]:
                raise RuntimeError("Curation changed before source binding repair")
            r.write_new(archive / "binding_before.json", {"rows": before, "protected": protected, "scope": scope})
            changed = c.execute("""UPDATE processed_evidence_candidate e SET
                raw_asset_id=d.source_record_id,
                source_record=jsonb_set(e.source_record,'{raw_asset_id}',to_jsonb(d.source_record_id)),
                metadata_json=e.metadata_json || jsonb_build_object('original_raw_asset_id_before_recovery',e.raw_asset_id),
                updated_at=now()
                FROM raw_document d WHERE e.raw_document_id=d.raw_document_id
                  AND e.evidence_id::text=ANY(%s) AND d.raw_document_id::text=ANY(%s)
                  AND e.raw_asset_id IS DISTINCT FROM d.source_record_id RETURNING e.evidence_id""",
                (scope["evidence_ids"], scope["new_raw_ids"])).fetchall()
            r.assert_protected(protected, snapshot(c, scope, True))
            verify(c, bundle)
            r.write_new(archive / "binding_after.json", {"rows": snapshot(c, scope)})
        r.write_new(archive / "binding_committed.json", {"status": "committed_and_verified", "rebound": len(changed),
                    "before_sha256": r.sha(archive / "binding_before.json"), "after_sha256": r.sha(archive / "binding_after.json")})
        print(json.dumps({"source_bindings_repaired": len(changed)}))


def rollback():
    """Restore exact before-images only while the scoped revision is unchanged."""
    import psycopg
    from psycopg import sql
    archive = curate.BASE / "transaction"
    before = curate.read(archive / "before.json")
    latest = archive / "binding_after.json" if (archive / "binding_committed.json").exists() else archive / "after.json"
    after = curate.read(latest)
    scope = before["scope"]
    with psycopg.connect(r.imp.psycopg_dsn(r.registered_url()), autocommit=True) as c:
        with c.transaction():
            r.lock_project_tables(c)
            if snapshot(c, scope) != after["rows"]:
                raise RuntimeError("Scoped evidence changed after curation; refuse to overwrite later work")
            protected = snapshot(c, scope, True)
            for table in ("rag_retrieval_record", "evidence_layer_membership", "enzyme_transporter_relationship"):
                clause, params = where(table, scope)
                c.execute(sql.SQL("DELETE FROM {} WHERE {}").format(sql.Identifier(table), sql.SQL(clause)), params)
            for table in ("processed_evidence_candidate", *[t for t in r.TYPED if t != "public_pbpk_model_registry"],
                          "evidence_layer_membership", "rag_retrieval_record", "enzyme_transporter_relationship"):
                r.restore_rows(c, table, before["rows"][table])
            c.execute("DELETE FROM raw_document WHERE raw_document_id::text=ANY(%s)", (scope["new_raw_ids"],))
            if snapshot(c, scope) != before["rows"]:
                raise RuntimeError("Rollback did not restore exact original rows")
            r.assert_protected(protected, snapshot(c, scope, True))
        r.write_new(archive / "rolled_back.json", {"status": "restored_exact_before_images", "before_sha256": r.sha(archive / "before.json")})
        print("Restored exact scoped before-images; other rows unchanged")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "apply", "repair-source-bindings", "rollback"])
    args = parser.parse_args()
    {"prepare": prepare, "apply": apply, "repair-source-bindings": repair_source_bindings, "rollback": rollback}[args.action]()
