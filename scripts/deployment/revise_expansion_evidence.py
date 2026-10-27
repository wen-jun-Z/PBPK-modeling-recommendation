"""Transactional, reversible serving revision for the frozen seven expansion runs.

Default action is read-only dry-run. This tool never invokes importer.run(),
changes schemas, replaces the whole database, or rewrites source/experiment files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from pbpk_kb.storage import import_processed_evidence as imp
from pbpk_kb.storage.expansion_identity_mapping import load_expansion_identity_mapping, lookup_registered_identities

ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT / "runtime/deployment/evidence_expansion_20261009/recovery_quality_20261014"
RUNS = frozenset(f"run_20261009T1409{n:02d}Z" for n in range(19, 26))
TYPED = {"physicochemical_property_evidence": imp.PHYS_COLUMNS, "adme_parameter_evidence": imp.ADME_COLUMNS,
         "clinical_pk_evidence": imp.CLINICAL_COLUMNS, "ddi_evidence": imp.DDI_COLUMNS,
         "special_population_evidence": imp.SPECIAL_COLUMNS, "tissue_expression_evidence": imp.TISSUE_COLUMNS,
         "public_pbpk_model_registry": imp.MODEL_COLUMNS}
PARENTS = {"raw_document": "raw_document_id", "parsed_document": "parsed_document_id",
           "protein_target_identity": "protein_target_id"}
PK = {"processed_evidence_candidate": ("evidence_id",),
      **{t: ("model_id" if t == "public_pbpk_model_registry" else "evidence_id",) for t in TYPED},
      "evidence_compound_attribution": ("attribution_id",), "evidence_layer_membership": ("evidence_id", "evidence_layer"),
      "evidence_duplicate_group": ("duplicate_group_id",), "evidence_duplicate_member": ("duplicate_group_id", "evidence_id"),
      "evidence_conflict_group": ("conflict_group_id",), "evidence_conflict_member": ("conflict_group_id", "evidence_id"),
      "rag_retrieval_record": ("retrieval_record_id",), "enzyme_transporter_relationship": ("relationship_id",),
      **{t: (key,) for t, key in PARENTS.items()}, "compound_identity": ("compound_id",),
      "public_pbpk_model_version": ("model_version_id",), "model_parameter_evidence": ("model_parameter_evidence_id",)}
INSERT_ORDER = ["processed_evidence_candidate", *TYPED, "evidence_compound_attribution", "evidence_layer_membership",
                "evidence_duplicate_group", "evidence_duplicate_member", "evidence_conflict_group", "evidence_conflict_member",
                "rag_retrieval_record", "enzyme_transporter_relationship"]
DELETE_ORDER = [t for t in reversed(INSERT_ORDER) if t not in {"processed_evidence_candidate", "public_pbpk_model_registry"}]


def inside(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise RuntimeError("Revision input/output escapes project")
    return path


def check_run(run: str) -> None:
    if run not in RUNS:
        raise RuntimeError("Only the seven frozen October expansion runs may be revised")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: Any) -> str:
    inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return sha(path)


def input_fingerprints(raw: Path, final: Path) -> dict[str, str]:
    paths = imp.raw_processing_layout(raw)
    files = [paths.path(name) for name in ("raw_inventory.jsonl", "parsed_documents.jsonl", "evidence_candidates.jsonl")]
    files += [p for p in final.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".npy"}]
    return {str(inside(p)): sha(p) for p in sorted(set(files))}


class RevisionRows(imp.EvidenceImporter):
    """Reuse importer transformations while every writer is an in-memory sink."""
    def __init__(self, run: str, raw: Path, final: Path, mapping):
        self.run_id = run
        self.run_dir = raw
        self.paths = imp.raw_processing_layout(raw)
        self.final_evidence_dir = final
        self.import_identity_mapping = mapping
        self.native_database_url = None  # Prevent identity enrichment/extra connections.
        self.limit = None
        self.replace = False
        self.counts = Counter()
        self.compound_ids = {}
        self.seen_candidate_ids = set()
        self.seen_typed_ids = {}
        self.tables: dict[str, dict[str, Any]] = {}
        self.final_attributions = self._load_final_attributions()
        self.auto_validated_rows = self._load_auto_validated_rows()
        self.serving_ids = set(self.auto_validated_rows)
        if not self.serving_ids:
            raise RuntimeError("Empty serving revision requires separate review; refusing automatic retirement of a whole batch")
        quarantined = {str(r["evidence_id"]) for r in imp.read_jsonl(final / "07_llm_quality_validation/final_quarantined_evidence.jsonl")}
        if self.serving_ids & quarantined:
            raise RuntimeError("Accepted evidence is also in the final quarantine partition")
        from pbpk_kb.processing.final_evidence_quality_stage import evaluate_domain_gate
        from pbpk_kb.processing.source_context import analyte_mismatch_reason
        for row in self.auto_validated_rows.values():
            if (row.get("fatal_blockers") or row.get("domain_gate_status") in {"excluded", "needs_manual_review"}
                    or evaluate_domain_gate(row)["domain_gate_status"] != "passed" or analyte_mismatch_reason(row)):
                raise RuntimeError("Serving partition still contains source/domain-review blockers")
        self.serving_chunk_ids = {str(r["chunk_id"]) for r in self.auto_validated_rows.values() if r.get("chunk_id")}
        self.linked_raw_asset_ids, self.linked_parsed_document_ids = self._load_linked_document_ids()
        self.raw_inventory_by_id = self._load_raw_inventory_by_id()

    def _exec_sql(self, sql):
        raise RuntimeError("Unexpected SQL during read-only row collection")

    def _copy_rows(self, table, columns, rows):
        if table not in PK:
            raise RuntimeError("Unapproved revision table")
        batch = self.tables.setdefault(table, {"columns": list(columns), "rows": []})
        if batch["columns"] != list(columns):
            raise RuntimeError("Inconsistent revision columns")
        for row in rows:
            mapped = self._map_identity_row(row)
            allowed = set(self.compound_ids.values())
            if any(mapped.get(key) is not None and str(mapped[key]) not in allowed
                   for key in ("compound_id", "attributed_compound_id", "victim_compound_id", "perpetrator_compound_id")):
                raise RuntimeError("Generated row refers to a compound outside this frozen batch")
            batch["rows"].append(mapped)

    def collect(self):
        self._require_files()
        for identity in imp.read_jsonl(self.final_evidence_dir / "01_compound_attribution/compound_identity.jsonl"):
            bound = self._map_identity_row(identity)
            self.compound_ids[str(identity["preferred_name"])] = str(bound["compound_id"])
        self._import_raw_documents()
        self._import_parsed_documents()
        for row in imp.read_jsonl(self.paths.path("evidence_candidates.jsonl")):
            eid = str(row.get("evidence_id") or "")
            if eid not in self.serving_ids:
                continue
            if row.get("run_id") != self.run_id or self.auto_validated_rows[eid].get("run_id") != self.run_id:
                raise RuntimeError("Candidate run differs from frozen revision scope")
            if eid in self.seen_candidate_ids:
                raise RuntimeError("Duplicate accepted candidate input")
            normalized = self._normalize_evidence_row({**row, **self.auto_validated_rows[eid]})
            final_id = self.auto_validated_rows[eid].get("attributed_compound_id") or self.auto_validated_rows[eid].get("compound_id")
            if str(normalized.get("compound_id")) != str(final_id):
                raise RuntimeError("Final accepted identity differs from final attribution")
            normalized["curation_status"] = "auto_validated"
            self._copy_rows("processed_evidence_candidate", imp.CANDIDATE_COLUMNS, [self._candidate_row(normalized)])
            typed = self._typed_row(normalized)
            if typed is not None:
                table, payload = typed
                self._copy_rows(table, TYPED[table], [payload])
            self.seen_candidate_ids.add(eid)
        if self.seen_candidate_ids != self.serving_ids:
            raise RuntimeError("Accepted evidence missing from candidate input")
        self._import_final_evidence_metadata()
        self._import_graph_relationships()
        for table, batch in self.tables.items():
            seen = set()
            for row in batch["rows"]:
                key = tuple(str(row.get(k)) for k in PK[table])
                if key in seen:
                    raise RuntimeError("Duplicate revision key: " + table)
                seen.add(key)
                if "run_id" in row and row["run_id"] != self.run_id:
                    raise RuntimeError("Generated row escaped run scope")
        return self


def registered_url():
    from verify_expansion_batch import _registered_database
    url, _ = _registered_database()
    return url.render_as_string(hide_password=False)


def collect(run: str, url: str, *, final_dir: Path | None = None) -> tuple[RevisionRows, dict[str, str]]:
    check_run(run)
    raw = inside(RECOVERY / "reprocessed" / run / "raw")
    final = inside(final_dir if final_dir is not None else RECOVERY / "reprocessed" / run / "final")
    if not final.is_relative_to(RECOVERY):
        raise RuntimeError("Serving revision must use an isolated recovery directory")
    if not raw.is_dir() or not final.is_dir():
        raise RuntimeError("Isolated reprocessed raw/final directories are not ready")
    before = input_fingerprints(raw, final)
    mapping = load_expansion_identity_mapping(ROOT / "data_process/final_evidence" / run,
        ROOT / "data_process/raw_processing" / run, lambda ids: lookup_registered_identities(url, ids))
    identity_file = Path("01_compound_attribution/compound_identity.jsonl")
    def identities(directory):
        return {str(row["preferred_name"]): str(mapping.record(row)["compound_id"])
                for row in imp.read_jsonl(directory / identity_file)}
    if identities(final) != identities(ROOT / "data_process/final_evidence" / run):
        raise RuntimeError("Reprocessed compound identities differ from the frozen original batch")
    rows = RevisionRows(run, raw, final, mapping).collect()
    if input_fingerprints(raw, final) != before:
        raise RuntimeError("Revision artifacts changed while being read")
    return rows, before


def scope_for(connection, rows: RevisionRows) -> dict[str, Any]:
    run = rows.run_id
    old = {str(r[0]) for r in connection.execute("SELECT evidence_id FROM processed_evidence_candidate WHERE run_id=%s", (run,))}
    proposed = rows.serving_ids
    conflicts = list(connection.execute("SELECT evidence_id FROM processed_evidence_candidate WHERE evidence_id::text=ANY(%s) AND run_id<>%s",
                                        (sorted(proposed), run)))
    if conflicts:
        raise RuntimeError("Revision evidence ID belongs to another run")
    ids = sorted(old | proposed)
    duplicate = {str(r[0]) for r in connection.execute("SELECT duplicate_group_id FROM evidence_duplicate_member WHERE evidence_id::text=ANY(%s)", (ids,))}
    duplicate |= {str(r[0]) for r in connection.execute("SELECT duplicate_group_id FROM evidence_duplicate_group WHERE representative_evidence_id::text=ANY(%s)", (ids,))}
    duplicate |= {str(r["duplicate_group_id"]) for r in rows.tables.get("evidence_duplicate_group", {}).get("rows", [])}
    conflict = {str(r[0]) for r in connection.execute("SELECT conflict_group_id FROM evidence_conflict_group WHERE run_id=%s", (run,))}
    conflict |= {str(r["conflict_group_id"]) for r in rows.tables.get("evidence_conflict_group", {}).get("rows", [])}
    if connection.execute("SELECT 1 FROM evidence_conflict_group WHERE conflict_group_id::text=ANY(%s) AND run_id<>%s LIMIT 1",
                          (sorted(conflict), run)).fetchone():
        raise RuntimeError("Revision conflict group belongs to another run")
    if connection.execute("SELECT 1 FROM evidence_duplicate_group WHERE duplicate_group_id::text=ANY(%s) "
                          "AND representative_evidence_id IS NOT NULL AND NOT(representative_evidence_id::text=ANY(%s)) LIMIT 1",
                          (sorted(duplicate), ids)).fetchone():
        raise RuntimeError("Revision duplicate representative belongs to protected evidence")
    for table, group, groups in [("evidence_duplicate_member", "duplicate_group_id", duplicate),
                                  ("evidence_conflict_member", "conflict_group_id", conflict)]:
        from psycopg import sql
        if connection.execute(sql.SQL("SELECT 1 FROM {} WHERE {}::text=ANY(%s) AND NOT(evidence_id::text=ANY(%s)) LIMIT 1")
                              .format(sql.Identifier(table), sql.Identifier(group)), (sorted(groups), ids)).fetchone():
            raise RuntimeError("Revision would alter a group shared with protected evidence")
    for table in TYPED:
        from psycopg import sql
        key = PK[table][0]
        if connection.execute(sql.SQL("SELECT 1 FROM {} WHERE {}::text=ANY(%s) LIMIT 1").format(sql.Identifier(table), sql.Identifier(key)),
                              (sorted(proposed - old),)).fetchone():
            raise RuntimeError("New candidate collides with an existing typed/model record")
    for table in ("public_pbpk_model_version", "model_parameter_evidence"):
        from psycopg import sql
        if connection.execute(sql.SQL("SELECT 1 FROM {} WHERE model_id::text=ANY(%s) LIMIT 1").format(sql.Identifier(table)), (ids,)).fetchone():
            raise RuntimeError("A scoped model record is used by model experiments; requires separate reviewed migration")
    if connection.execute("SELECT 1 FROM rag_retrieval_record r WHERE run_id<>%s AND EXISTS "
                          "(SELECT 1 FROM jsonb_array_elements_text(r.evidence_ids) x WHERE x=ANY(%s)) LIMIT 1", (run, ids)).fetchone():
        raise RuntimeError("Protected retrieval record links to revision evidence")
    existing_compounds = {str(r[0]) for r in connection.execute("SELECT compound_id FROM compound_identity")}
    if not set(rows.compound_ids.values()) <= existing_compounds:
        raise RuntimeError("Revision cannot create or change registered compound identities")
    parents_new = {}
    for table, key in PARENTS.items():
        from psycopg import sql
        proposed_keys = {str(r[key]) for r in rows.tables.get(table, {}).get("rows", [])}
        existing = {str(r[0]) for r in connection.execute(sql.SQL("SELECT {} FROM {} WHERE {}::text=ANY(%s)")
                    .format(sql.Identifier(key), sql.Identifier(table), sql.Identifier(key)), (sorted(proposed_keys),))}
        parents_new[table] = sorted(proposed_keys - existing)
        verify_existing_parent_identity(connection, table, rows.tables.get(table, {}).get("rows", []), existing)
    return {"run_id": run, "evidence_ids": ids, "old_ids": sorted(old), "serving_ids": sorted(proposed),
            "duplicate_ids": sorted(duplicate), "conflict_ids": sorted(conflict), "parents_new": parents_new}


def verify_existing_parent_identity(connection, table, proposed, existing):
    from psycopg import sql
    fields = {"raw_document": ("source_id", "raw_checksum", "raw_storage_path"),
              "parsed_document": ("raw_document_id", "source_id", "source_location", "parsed_checksum"),
              "protein_target_identity": ("gene_symbol", "target_type")}[table]
    if not existing:
        return
    key = PARENTS[table]
    stored = {str(row[key]): row for (row,) in connection.execute(sql.SQL("SELECT to_jsonb(t) FROM {} t WHERE {}::text=ANY(%s)")
                    .format(sql.Identifier(table), sql.Identifier(key)), (sorted(existing),))}
    for row in proposed:
        prior = stored.get(str(row[key]))
        if prior is not None and any(str(prior.get(field)) != str(row.get(field)) for field in fields):
            raise RuntimeError("Existing parent identity/source bytes differ: " + table)


def selection(table: str, scope: dict) -> tuple[str, list]:
    if table in PARENTS:
        return PARENTS[table] + "::text=ANY(%s)", [scope["parents_new"][table]]
    if table in {"compound_identity", "public_pbpk_model_version", "model_parameter_evidence"}:
        return "FALSE", []
    if table == "rag_retrieval_record":
        return "run_id=%s", [scope["run_id"]]
    if table == "evidence_duplicate_group":
        return "duplicate_group_id::text=ANY(%s)", [scope["duplicate_ids"]]
    if table == "evidence_conflict_group":
        return "conflict_group_id::text=ANY(%s)", [scope["conflict_ids"]]
    key = "processed_evidence_id" if table == "enzyme_transporter_relationship" else "model_id" if table == "public_pbpk_model_registry" else "evidence_id"
    return key + "::text=ANY(%s)", [scope["evidence_ids"]]


def snapshot(connection, scope: dict, *, hashes: bool = False, protected: bool = False) -> dict:
    from psycopg import sql
    result = {}
    for table in PK:
        where, args = selection(table, scope)
        if protected:
            where = "NOT COALESCE((" + where + "), FALSE)"
        value = "md5(to_jsonb(t)::text)" if hashes else "to_jsonb(t)::text"
        order = sql.SQL(",").join(sql.Identifier(key) for key in PK[table])
        query = sql.SQL("SELECT {} FROM {} t WHERE {} ORDER BY {}").format(sql.SQL(value), sql.Identifier(table), sql.SQL(where), order)
        result[table] = [r[0] for r in connection.execute(query, args)]
    return result


def assert_protected(before: dict, after: dict) -> None:
    changed = [table for table in before if before[table] != after.get(table)]
    if changed:
        raise RuntimeError("Protected rows changed; revision must roll back: " + ", ".join(changed))


def copy_rows(connection, table, columns, rows, *, upsert=False, ignore_existing=False):
    if not rows:
        return
    from psycopg import sql
    temp = "revision_" + uuid4().hex
    connection.execute(sql.SQL("CREATE TEMP TABLE {} (LIKE {} INCLUDING DEFAULTS) ON COMMIT DROP").format(sql.Identifier(temp), sql.Identifier(table)))
    cols = sql.SQL(",").join(sql.Identifier(c) for c in columns)
    with connection.cursor() as cursor:
        with cursor.copy(sql.SQL("COPY {} ({}) FROM STDIN").format(sql.Identifier(temp), cols)) as writer:
            for row in rows:
                writer.write_row([imp.csv_value(row.get(c)) for c in columns])
    suffix = sql.SQL("")
    if upsert:
        updates = sql.SQL(",").join(sql.SQL("{}=EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c)) for c in columns if c not in PK[table])
        if table in {"processed_evidence_candidate", "public_pbpk_model_registry"}:
            updates += sql.SQL(",updated_at=now()")
        suffix = sql.SQL(" ON CONFLICT ({}) DO UPDATE SET {}").format(sql.SQL(",").join(map(sql.Identifier, PK[table])), updates)
    elif ignore_existing:
        suffix = sql.SQL(" ON CONFLICT DO NOTHING")
    connection.execute(sql.SQL("INSERT INTO {} ({}) SELECT {} FROM {}{}").format(sql.Identifier(table), cols, cols, sql.Identifier(temp), suffix))


def clear_children(connection, scope):
    from psycopg import sql
    for table in DELETE_ORDER:
        where, args = selection(table, scope)
        connection.execute(sql.SQL("DELETE FROM {} WHERE {}").format(sql.Identifier(table), sql.SQL(where)), args)


def apply_rows(connection, rows: RevisionRows, scope: dict, revision: str):
    for table in PARENTS:
        batch = rows.tables.get(table)
        if batch:
            copy_rows(connection, table, batch["columns"], batch["rows"], ignore_existing=True)
    clear_children(connection, scope)
    for table in INSERT_ORDER:
        batch = rows.tables.get(table)
        if batch:
            copy_rows(connection, table, batch["columns"], batch["rows"],
                      upsert=table in {"processed_evidence_candidate", "public_pbpk_model_registry"})
    retired = sorted(set(scope["old_ids"]) - rows.serving_ids)
    if retired:
        note = json.dumps({"revision_id": revision, "reason": "not_accepted_after_source_revalidation"})
        connection.execute("UPDATE processed_evidence_candidate SET curation_status='deprecated', updated_at=now(), "
                           "metadata_json=metadata_json || jsonb_build_object('revision_exclusion',%s::jsonb) WHERE evidence_id::text=ANY(%s)", (note, retired))
        connection.execute("UPDATE public_pbpk_model_registry SET curation_status='deprecated',updated_at=now() WHERE model_id::text=ANY(%s)", (retired,))
        connection.execute("INSERT INTO evidence_layer_membership(evidence_id,run_id,evidence_layer,exclusion_reasons,metadata_json) "
                           "SELECT evidence_id,run_id,'excluded','[\"not_accepted_after_source_revalidation\"]'::jsonb,%s::jsonb "
                           "FROM processed_evidence_candidate WHERE evidence_id::text=ANY(%s)", (note, retired))


def verify_serving(connection, scope):
    ids = {str(r[0]) for r in connection.execute("SELECT evidence_id FROM auto_validated_evidence WHERE run_id=%s", (scope["run_id"],))}
    if ids != set(scope["serving_ids"]):
        raise RuntimeError("Serving evidence differs from revised accepted partition")
    linked = {r[0] for r in connection.execute("SELECT DISTINCT jsonb_array_elements_text(evidence_ids) FROM rag_retrieval_record WHERE run_id=%s", (scope["run_id"],))}
    if linked != ids:
        raise RuntimeError("RAG is missing accepted evidence or still links retired evidence")


def lock_project_tables(connection):
    from psycopg import sql
    if not connection.execute("SELECT pg_try_advisory_xact_lock(hashtext('pbpk_expansion_quality_revision_20261014'))").fetchone()[0]:
        raise RuntimeError("Another project evidence revision owns the transaction lock")
    connection.execute(sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(sql.SQL(",").join(sql.Identifier(t) for t in sorted(PK))))


def revise(run: str, *, apply: bool = False, final_dir: Path | None = None) -> dict:
    check_run(run)
    url = registered_url()
    rows, inputs = collect(run, url) if final_dir is None else collect(run, url, final_dir=final_dir)
    import psycopg
    options = "-c statement_timeout=120000 -c lock_timeout=10000" + ("" if apply else " -c default_transaction_read_only=on")
    with psycopg.connect(imp.psycopg_dsn(url), autocommit=True, options=options) as connection:
        with connection.transaction():
            if apply:
                lock_project_tables(connection)
            scope = scope_for(connection, rows)
            protected = snapshot(connection, scope, hashes=True, protected=True)
            summary = {"run_id": run, "action": "apply" if apply else "dry-run", "old_count": len(scope["old_ids"]),
                       "accepted_count": len(rows.serving_ids), "added": len(rows.serving_ids - set(scope["old_ids"])),
                       "retired": len(set(scope["old_ids"]) - rows.serving_ids),
                       "counts": {table: len(batch["rows"]) for table, batch in rows.tables.items()},
                       "identity_mapping": rows.import_identity_mapping.receipt}
            if not apply:
                return {**summary, "database_changed": False}
            revision = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
            audit = inside(RECOVERY / "db_revisions" / revision)
            before = {"scope": scope, "rows": snapshot(connection, scope), "protected": protected,
                      "inputs": inputs, "summary": summary}
            before_sha = write_new(audit / "before.json", before)
            if input_fingerprints(rows.run_dir, rows.final_evidence_dir) != inputs:
                raise RuntimeError("Reprocessed files changed before transaction")
            apply_rows(connection, rows, scope, revision)
            verify_serving(connection, scope)
            assert_protected(protected, snapshot(connection, scope, hashes=True, protected=True))
            after_sha = write_new(audit / "after.json", {"scope": scope, "hashes": snapshot(connection, scope, hashes=True)})
            write_new(audit / "transaction.json", {"before_sha256": before_sha, "after_sha256": after_sha,
                                                   "state": "prepared_for_commit", "revision_id": revision})
        # The before/after snapshots exist before COMMIT. If interrupted here,
        # rollback verifies the live after-hash rather than assuming a status.
        write_new(audit / "committed.json", {**summary, "revision_id": revision, "protected_rows_unchanged": True})
        return {**summary, "revision_id": revision, "audit_dir": str(audit), "protected_rows_unchanged": True}


def restore_rows(connection, table: str, records: list[str]):
    if not records:
        return
    from psycopg import sql
    columns = list(json.loads(records[0]))
    updates = sql.SQL(",").join(sql.SQL("{}=EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c)) for c in columns if c not in PK[table])
    connection.execute(sql.SQL("INSERT INTO {} SELECT * FROM jsonb_populate_recordset(NULL::{},%s::jsonb) "
                               "ON CONFLICT ({}) DO UPDATE SET {}").format(sql.Identifier(table), sql.Identifier(table),
                               sql.SQL(",").join(map(sql.Identifier, PK[table])), updates), ("[" + ",".join(records) + "]",))


def rollback(audit_dir: Path) -> dict:
    audit = inside(audit_dir)
    if not audit.is_relative_to(RECOVERY / "db_revisions"):
        raise RuntimeError("Unregistered revision archive")
    transaction = json.loads((audit / "transaction.json").read_text())
    if sha(audit / "before.json") != transaction["before_sha256"] or sha(audit / "after.json") != transaction["after_sha256"]:
        raise RuntimeError("Revision archive checksum mismatch")
    before = json.loads((audit / "before.json").read_text())
    after = json.loads((audit / "after.json").read_text())
    scope = before["scope"]
    check_run(scope["run_id"])
    if scope != after["scope"]:
        raise RuntimeError("Revision archive scope mismatch")
    if set(scope["evidence_ids"]) != set(scope["old_ids"]) | set(scope["serving_ids"]):
        raise RuntimeError("Revision archive has inconsistent evidence scope")
    if any(before["rows"][table] for table in PARENTS):
        raise RuntimeError("Rollback cannot remove pre-existing source/target parents")
    if any(json.loads(row).get("run_id") != scope["run_id"]
           for row in before["rows"]["processed_evidence_candidate"]):
        raise RuntimeError("Archived candidate belongs to another run")
    import psycopg
    from psycopg import sql
    with psycopg.connect(imp.psycopg_dsn(registered_url()), autocommit=True,
                         options="-c statement_timeout=120000 -c lock_timeout=10000") as connection:
        with connection.transaction():
            lock_project_tables(connection)
            if connection.execute("SELECT 1 FROM processed_evidence_candidate WHERE evidence_id::text=ANY(%s) AND run_id<>%s LIMIT 1",
                                  (scope["evidence_ids"], scope["run_id"])).fetchone():
                raise RuntimeError("Rollback evidence ID belongs to another run")
            if snapshot(connection, scope, hashes=True) != after["hashes"]:
                raise RuntimeError("Revision state changed after commit; rollback requires review")
            assert_protected(before["protected"], snapshot(connection, scope, hashes=True, protected=True))
            clear_children(connection, scope)
            new_ids = sorted(set(scope["serving_ids"]) - set(scope["old_ids"]))
            connection.execute("DELETE FROM public_pbpk_model_registry WHERE model_id::text=ANY(%s)", (new_ids,))
            connection.execute("DELETE FROM processed_evidence_candidate WHERE evidence_id::text=ANY(%s)", (new_ids,))
            for table in INSERT_ORDER:
                restore_rows(connection, table, before["rows"][table])
            for table in reversed(PARENTS):
                connection.execute(sql.SQL("DELETE FROM {} WHERE {}::text=ANY(%s)")
                                   .format(sql.Identifier(table), sql.Identifier(PARENTS[table])), (scope["parents_new"][table],))
            if snapshot(connection, scope) != before["rows"]:
                raise RuntimeError("Rollback did not restore exact archived rows")
            assert_protected(before["protected"], snapshot(connection, scope, hashes=True, protected=True))
        write_new(audit / "rolled_back.json", {"run_id": scope["run_id"], "restored": True})
    return {"run_id": scope["run_id"], "restored": True, "protected_rows_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["dry-run", "apply", "rollback"], nargs="?", default="dry-run")
    parser.add_argument("--run-id")
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--final-dir", type=Path, help="Use a separately verified final partition; preserve earlier artifacts")
    args = parser.parse_args()
    result = rollback(args.audit_dir) if args.action == "rollback" else revise(args.run_id, apply=args.action == "apply", final_dir=args.final_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
