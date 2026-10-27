"""Bounded Qwen repair + separate validation of remaining source-backed context.

Uses the existing quality pipeline. Source failures and unresolved species are
not converted into positives. Each record has at most one completed review;
three consecutive service failures stop the worker. Promotion is transactional.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import signal
import time
from uuid import NAMESPACE_URL, uuid5

import apply_legacy_curation as apply_tool
import curate_legacy_evidence as curate
import revise_expansion_evidence as r
from verify_expansion_batch import _registered_database, _local_query_vectors
from pbpk_kb.processing.auto_quality_llm_stage import AutoQualityLLMStage, QualityOutputError
from pbpk_kb.processing.evidence_llm import EvidenceServerOutputError, EvidenceGenerationError
from pbpk_kb.processing.final_evidence_quality_stage import evaluate_domain_gate
from pbpk_kb.processing.auto_quality_stage import evaluate_evidence
from pbpk_kb.processing.rag_index_stage import evidence_record, build_embedding_text

OUT = curate.BASE / "context_followup"
REPAIRABLE = {"relative_effect_missing_reference", "quantitative_statement_missing_value", "numeric_value_missing_unit",
              "ddi_missing_counterparty_or_mechanism", "numeric_value_belongs_to_other_substance", "numeric_value_not_supported_for_target_parameter"}
STOP_REQUESTED = False


class SourceContextError(ValueError):
    """A deterministic missing/ambiguous source is not a service outage."""


def input_rows():
    import psycopg
    proposal = curate.read(curate.BASE / "curation_proposal.json")
    selected = {x["evidence_id"]: x for x in proposal["items"] if x["status"] == "reviewable" and set(x["blockers"]) <= REPAIRABLE}
    with psycopg.connect(r.imp.psycopg_dsn(r.registered_url()), options="-c default_transaction_read_only=on") as c:
        rows = [x[0] for x in c.execute("SELECT to_jsonb(t) FROM reviewable_evidence t WHERE evidence_id::text=ANY(%s)", (sorted(selected),))]
        c.rollback()
    if {x["evidence_id"] for x in rows} != set(selected):
        raise RuntimeError("Pending context queue differs from serving curation")
    return [{**row, "repairable_blockers": selected[row["evidence_id"]]["blockers"]} for row in rows]


def context(row):
    path = r.inside(Path(row["storage_path"]))
    if r.sha(path) != row["checksum"]:
        raise RuntimeError("Source changed before context review")
    quote = row["source_text"]
    content = curate.source_content(path)
    if row["source_id"] in {"pubmed", "pmc_oa"} and path.suffix.lower() in {".xml", ".nxml"}:
        from defusedxml import ElementTree as ET
        from pbpk_kb.parsers.table_structure import element_text, local_name
        root = ET.fromstring(path.read_bytes())
        articles = [curate.normalize(element_text(n)) for n in root.iter() if local_name(n.tag) in {"article", "PubmedArticle"}]
        matching = list(dict.fromkeys(t for t in articles if curate.normalize(quote) in t))
        if len(matching) != 1:
            raise SourceContextError("Source quote has no unique article context")
        content["texts"] = matching
    normalized = curate.normalize(quote)
    matches = []
    for text in content["texts"]:
        start = text.find(normalized)
        if start >= 0:
            # At most neighbouring sentences; never an unrelated full article
            # or another item in a batched PubMed response.
            left = max(text.rfind(". ", 0, max(0, start-1)) + 2, start-600, 0)
            right = text.find(". ", start+len(normalized))
            right = min(right+1 if right >= 0 else len(text), start+len(normalized)+800)
            matches.append(text[left:right])
    if not matches:
        raise SourceContextError("Current quote cannot be located in recovered source")
    if len(set(matches)) != 1:
        raise SourceContextError("Source quote has ambiguous neighbouring context")
    selected = matches[0]
    combined = quote + "\n\n" + selected
    if len(combined) > 15000:
        raise SourceContextError("Source context exceeds bounded review input")
    return {"chunks": [{"source_location": row["source_location"], "text": quote},
                       {"source_location": row["source_location"], "text": selected}], "combined_text": combined,
            "source_checksum": row["checksum"], "source_path": str(path)}


def run():
    def stop_after_record(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
    signal.signal(signal.SIGTERM, stop_after_record)
    signal.signal(signal.SIGINT, stop_after_record)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows_path = OUT / "input.json"
    if rows_path.exists():
        rows = curate.read(rows_path)
    else:
        rows = input_rows(); r.write_new(rows_path, rows)
    stage = AutoQualityLLMStage(final_run_dir=OUT, raw_run_dir=OUT, workers=1)
    # This worker repairs one small record at a time. Long repeated quotes do
    # not improve validation; incomplete output is still rejected, not parsed
    # or auto-accepted. This setting is local to this worker process.
    import pbpk_kb.processing.evidence_llm as evidence_llm
    evidence_llm.MAX_EVIDENCE_OUTPUT_TOKENS = 1536
    curate.atomic(OUT / "protocol.json", {"model": stage.model, "workers": 1, "output_tokens": 1536,
                   "source_bound": True, "second_validation_call": True, "independent_expert_validation": False})
    results, streak = [], 0
    curate.atomic(OUT / "status.json", {"status": "running", "done": 0, "total": len(rows), "counts": {}})
    for original in rows:
        if STOP_REQUESTED or (OUT / "stop_requested").exists():
            curate.atomic(OUT / "status.json", {"status": "paused_at_checkpoint", "done": len(results), "total": len(rows), "counts": dict(Counter(x["status"] for x in results))})
            return
        if shutil.disk_usage(OUT).free < 12 * 1024**3:
            raise RuntimeError("Insufficient project free space for context review")
        path = OUT / (original["evidence_id"] + ".json")
        if path.exists():
            result = curate.read(path)
            if result.get("terminal"):
                results.append(result)
                curate.atomic(OUT / "status.json", {"status": "running", "done": len(results), "total": len(rows),
                              "counts": dict(Counter(x["status"] for x in results))})
                continue
        row = {**original.get("source_record", {}), **{k: v for k, v in original.items() if k != "source_record"}}
        for field in ("value_low", "value_high"):
            if row.get(field) is not None:
                row[field] = float(row[field])
        try:
            source = context(row)
            row["repairable_blockers"] = [*row.get("repairable_blockers", []),
                "For a relative effect, reference means the explicitly described comparison regimen/population, not a citation, URL, PMID, or source filename."]
            result = stage._process_one(row, source)
            if result["status"] == "auto_validated":
                domain = evaluate_domain_gate(result["evidence"])
                from repair_legacy_source_gaps import endpoint_semantics_blockers
                extra = endpoint_semantics_blockers(result["evidence"])
                reference = str((result["evidence"].get("metadata_json") or {}).get("reference") or "")
                if re.search(r"https?://|\b(?:pubmed|pmc|doi|pmid):|\.(?:xml|pdf|json)\b", reference, re.I):
                    extra.append("citation_is_not_comparison_regimen")
                if domain["domain_gate_status"] != "passed" or extra:
                    result.update(status="quarantined_domain_review", domain_review=domain, extra_blockers=extra)
            result["source_context_sha256"] = apply_tool.hash_object(source)
            result["source_checksum"] = row["checksum"]
            streak = 0
        except SourceContextError as exc:
            result = {"evidence_id": row["evidence_id"], "terminal": True, "status": "quarantined_source_context", "reason": str(exc)}
            streak = 0
        except (EvidenceServerOutputError, EvidenceGenerationError, QualityOutputError) as exc:
            # A bounded, invalid generation is a review outcome, not evidence
            # support and not proof that the service is down. Keep it separate
            # from source failures; never import or salvage partial JSON.
            result = {"evidence_id": row["evidence_id"], "terminal": True,
                      "status": "quarantined_incomplete_model_output", "error_type": type(exc).__name__}
            streak = 0
        except Exception as exc:
            result = {"evidence_id": row["evidence_id"], "terminal": False, "status": "failed_review", "error_type": type(exc).__name__}
            streak += 1
        result["input_sha256"] = apply_tool.hash_object(original)
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        curate.atomic(path, result)
        results.append(result)
        counts = dict(Counter(x["status"] for x in results))
        curate.atomic(OUT / "status.json", {"status": "running" if streak < 3 else "stopped_service_failures", "done": len(results), "total": len(rows), "counts": counts})
        print(json.dumps({"done": len(results), "total": len(rows), "status": result["status"]}), flush=True)
        if streak >= 3:
            raise RuntimeError("Three consecutive review failures; no automatic retry loop")
    promote(rows, results)
    curate.atomic(OUT / "status.json", {"status": "completed" if all(x.get("terminal") for x in results) else "completed_with_unresolved_service_errors", "done": len(results), "total": len(rows),
                  "counts": dict(Counter(x["status"] for x in results)), "promotion_receipt": str(OUT / "promoted.json")})


def promote(original_rows, results):
    import psycopg
    from psycopg.types.json import Jsonb
    accepted = [x for x in results if x["status"] == "auto_validated"]
    if not accepted:
        curate.atomic(OUT / "promoted.json", {"promoted": 0}); return
    if (OUT / "promoted.json").exists():
        return
    originals = {x["evidence_id"]: x for x in original_rows}
    rows = [x["evidence"] for x in accepted]
    def signature(row):
        fields = ("compound_id", "compound_role", "evidence_type", "parameter", "value", "unit", "value_low", "value_high", "value_qualifier", "condition", "population", "route", "dose", "species", "source_text")
        return apply_tool.hash_object([row.get(k) for k in fields])
    with psycopg.connect(r.imp.psycopg_dsn(r.registered_url()), options="-c default_transaction_read_only=on") as c:
        current = [x[0] for x in c.execute("SELECT to_jsonb(t) FROM auto_validated_evidence t")]
        c.rollback()
    known = {signature({**x.get("source_record", {}), **{k: v for k, v in x.items() if k != "source_record"}}): x["evidence_id"] for x in current}
    unique, duplicates = [], []
    for row in rows:
        key = signature(row)
        if key in known:
            duplicates.append({"evidence_id": row["evidence_id"], "canonical_evidence_id": known[key]})
        else:
            known[key] = row["evidence_id"]; unique.append(row)
    rows = unique
    if not rows:
        curate.atomic(OUT / "promoted.json", {"promoted": 0, "duplicates_not_readmitted": duplicates}); return
    for row in rows:
        from repair_legacy_source_gaps import endpoint_semantics_blockers
        if (evaluate_domain_gate(row)["domain_gate_status"] != "passed" or evaluate_evidence(row, [])["quality_status"] != "auto_validated"
                or endpoint_semantics_blockers(row)):
            raise RuntimeError("Promotion no longer satisfies independent source rules")
        if r.sha(r.inside(Path(row["storage_path"]))) != row["checksum"]:
            raise RuntimeError("Promotion source changed")
    records = [evidence_record(row, "auto_validated") for row in rows]
    for record in records:
        record["embedding_text"] = build_embedding_text(record)
        record["retrieval_record_id"] = str(uuid5(NAMESPACE_URL, "legacy_context_followup_20261022|" + record["evidence_ids"][0]))
    url, config = _registered_database()
    vectors, revision = _local_query_vectors([x["embedding_text"] for x in records], config)
    scope = {"evidence_ids": sorted(originals[r["evidence_id"]]["evidence_id"] for r in rows),
             "raw_ids": [], "rag_ids": [x["retrieval_record_id"] for x in records]}
    with psycopg.connect(r.imp.psycopg_dsn(url.render_as_string(hide_password=False)), autocommit=True,
                        options="-c statement_timeout=180000 -c lock_timeout=10000") as c:
        with c.transaction():
            r.lock_project_tables(c)
            current = {x["evidence_id"]: x for (x,) in c.execute("SELECT to_jsonb(t) FROM reviewable_evidence t WHERE evidence_id::text=ANY(%s)", (scope["evidence_ids"],))}
            if set(current) != set(scope["evidence_ids"]):
                raise RuntimeError("An evidence record changed serving layer during review")
            for eid, original in originals.items():
                if eid in current:
                    expected = {k: v for k, v in original.items() if k != "repairable_blockers"}
                    if apply_tool.hash_object(current[eid]) != apply_tool.hash_object(expected):
                        raise RuntimeError("A reviewed row changed during inference")
            before, protected = apply_tool.snapshot(c, scope), apply_tool.snapshot(c, scope, True)
            r.write_new(OUT / "before_promotion.json", {"scope": scope, "rows": before, "protected": protected})
            for row in rows:
                row = dict(row); eid = row["evidence_id"]
                row["curation_status"] = "auto_validated"
                row["metadata_json"] = {**row.get("metadata_json", {}), "source_context_followup": {"status": "validated", "receipt": str(OUT / (eid + ".json"))}}
                payload = r.imp.EvidenceImporter._candidate_row(None, row); payload.pop("evidence_id")
                for k in ("source_record", "metadata_json"): payload[k] = json.loads(payload[k])
                payload["updated_at"] = datetime.now(timezone.utc)
                apply_tool.update_fields(c, "processed_evidence_candidate", "evidence_id", eid, payload)
                if row["evidence_type"] != "public_pbpk_model":
                    table, typed = r.imp.EvidenceImporter._typed_row(None, row); typed.pop("evidence_id")
                    apply_tool.update_fields(c, table, "evidence_id", eid, typed)
                # The before-image retains the old review blockers. Current
                # serving and review queues must not both contain this row.
                c.execute("DELETE FROM evidence_layer_membership WHERE evidence_id::text=%s AND evidence_layer='reviewable'", (eid,))
                c.execute("INSERT INTO evidence_layer_membership(evidence_id,run_id,evidence_layer,metadata_json) VALUES (%s,%s,'auto_validated',%s)",
                          (eid, row["run_id"], Jsonb({"source_context_followup": "validated"})))
            indices = dict(c.execute("SELECT run_id,max(embedding_row)+1 FROM rag_retrieval_record GROUP BY run_id"))
            rag_rows = []
            for record in records:
                payload = {k: record.get(k) for k in r.imp.RAG_COLUMNS}
                payload.update(embedding_row=indices.get(record["run_id"], 0), embedding=r.imp.vector_text(vectors[record["embedding_text"]]),
                               embedding_model="NeuML/pubmedbert-base-embeddings", evidence_layer="auto_validated", curation_status="auto_validated",
                               provenance_status="validated", metadata_json=json.dumps({"source_context_followup": "validated"}))
                indices[record["run_id"]] = payload["embedding_row"] + 1
                for key in ("record_kinds", "evidence_ids", "chunk_ids", "parsed_table_ids", "recovered_table_ids"):
                    payload[key] = json.dumps(payload.get(key) or [])
                rag_rows.append(payload)
            r.copy_rows(c, "rag_retrieval_record", r.imp.RAG_COLUMNS, rag_rows)
            r.assert_protected(protected, apply_tool.snapshot(c, scope, True))
            count = c.execute("SELECT count(*) FROM auto_validated_evidence WHERE evidence_id::text=ANY(%s)", (scope["evidence_ids"],)).fetchone()[0]
            if count != len(rows): raise RuntimeError("Promoted evidence count differs")
            r.write_new(OUT / "after_promotion.json", {"rows": apply_tool.snapshot(c, scope)})
        r.write_new(OUT / "promoted.json", {"promoted": len(rows), "embedding_revision": revision, "duplicates_not_readmitted": duplicates,
                     "source_rule_and_domain_rechecked": True, "protected_rows_verified": sum(map(len, protected.values()))})


if __name__ == "__main__":
    run()
