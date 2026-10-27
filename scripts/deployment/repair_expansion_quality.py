"""Isolated, resumable reprocessing of the seven new-compound batches.

Nothing starts on import or with plan/status/watch. start and run require
--execute and a passed, code-bound pilot_v3-or-later receipt. Database revision
is dry-run unless --apply-validated is explicit. Acquisition and original
accepted outputs are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
BASE = ROOT / "runtime/deployment/evidence_expansion_20261009/recovery_quality_20261014"
RUNS = tuple(f"run_20261009T1409{n:02d}Z" for n in range(19, 26))
STAGES = (
    "prepare_inputs", "classify", "llm_extract", "table_recovery", "completeness",
    "source_context", "provenance_attribution", "domain_and_deterministic_dedup",
    "semantic_dedup", "conflict_and_evidence_layers", "rag_index",
    "deterministic_quality_control", "llm_quality_validation", "source_context_guard",
    "rag_audit", "verify_originals", "database_dry_run",
)
APPLY_STAGES = STAGES + ("database_apply", "verified_retrieval")
READ_ONLY_INPUTS = (
    "raw_inventory.jsonl", "raw_asset_aliases.jsonl", "parsed_documents.jsonl",
    "parsed_records.jsonl", "parsed_tables.jsonl", "archive_members.jsonl", "asset_links.jsonl",
    "text_chunks.jsonl", "structured_record_processing_status.jsonl",
    "structured_table_processing_status.jsonl", "evidence_pointers.jsonl",
)
COPY_INPUTS = (
    "processing_summary.json", "inventory_summary.json", "parse_summary.json",
    "deterministic_extract_summary.json", "acquisition_failure_status.jsonl",
    "asset_parse_status.jsonl", "processing_failures.jsonl", "chunk_processing_status.jsonl",
    "llm_extraction_requests_live.jsonl", "llm_extraction_responses_live.jsonl",
)
LLM_STAGES = {"classify", "llm_extract", "provenance_attribution", "llm_quality_validation"}
GIB = 1024 ** 3
MIB = 1024 ** 2
STOP = False


def stages_for(apply_validated: bool = False):
    return APPLY_STAGES if apply_validated else STAGES


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inside(path: Path | str, boundary: Path | None = None) -> Path:
    boundary = ROOT if boundary is None else boundary
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(boundary.resolve()):
        raise ValueError("Path escapes the authorized project/resource boundary")
    return resolved


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else ({} if default is None else default)


def atomic_json(path: Path, value) -> None:
    inside(path, BASE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def rows(path: Path):
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def atomic_rows(path: Path, values) -> int:
    inside(path, BASE)
    if path.is_symlink():
        raise ValueError("Refusing to write an input symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
            count += 1
    temporary.replace(path)
    return count


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * MIB), b""):
            value.update(block)
    return value.hexdigest()


def paths(run_id: str) -> dict[str, Path]:
    if run_id not in RUNS:
        raise ValueError("Only the seven new 20261009 batches can be reprocessed")
    destination = BASE / "reprocessed" / run_id
    return {"root": destination, "raw": destination / "raw", "final": destination / "final",
            "receipt": destination / "input_receipt.json", "state": destination / "controller_state.json",
            "original_raw": ROOT / "data_process/raw_processing" / run_id,
            "original_final": ROOT / "data_process/final_evidence" / run_id,
            "acquisition": ROOT / "artifacts/pbpk_raw_runs/compound_runs" / run_id}


def validate_pilot(path: Path | str) -> dict:
    path = inside(path, BASE)
    match = re.search(r"pilot_v(\d+)", str(path))
    if not match or int(match.group(1)) < 3:
        raise ValueError("A passed pilot_v3-or-later receipt is required before execution")
    receipt = read_json(path)
    if receipt.get("passed") is not True or not isinstance(receipt.get("results"), list):
        raise ValueError("The pilot receipt has not passed")
    hashes = receipt.get("code_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("The pilot must bind validated code SHA256 values")
    required = "src/pbpk_kb/processing/run_llm_extraction.py"
    if required not in hashes:
        raise ValueError("The pilot does not bind the current extraction implementation")
    for relative, expected in hashes.items():
        source = inside(relative)
        if not source.is_file() or digest(source) != expected:
            raise ValueError(f"Validated pilot code changed: {relative}")
    return {"path": str(path), "sha256": digest(path), "version": int(match.group(1)),
            "prompt": receipt.get("prompt"), "code_sha256": hashes}


def source_snapshot(source: Path) -> dict:
    source = inside(source)
    stat = source.stat()
    return {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest(source)}


def copy_file(source: Path, destination: Path) -> None:
    inside(source)
    inside(destination, BASE)
    if destination.is_symlink():
        raise ValueError("Refusing to overwrite an input symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".seed.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def seed_inputs(run_id: str) -> dict:
    from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
    selected = paths(run_id)
    if selected["receipt"].is_file():
        verify_originals(run_id, full=False)
        return {"accepted": True, "reused": True}
    old = raw_processing_layout(selected["original_raw"])
    new = raw_processing_layout(selected["raw"])
    original_state = read_json(old.path("pipeline_state.json"))
    if original_state.get("run_id") != run_id:
        raise ValueError("Original accepted raw state has the wrong run identity")
    for stage in ("inventory", "parse", "deterministic_extract", "llm_extract", "table_recovery", "completeness"):
        existing = (original_state.get("stages") or {}).get(stage) or {}
        if existing.get("accepted") is not True or existing.get("status") != "completed":
            raise ValueError(f"Original run is not fully accepted: {stage}")
    if not selected["original_final"].is_dir():
        raise ValueError("The original final evidence directory is missing")
    new.ensure_dirs()
    originals = {}
    symlinks = {}
    for filename in READ_ONLY_INPUTS:
        source = old.path(filename)
        if not source.is_file():
            if filename in {"raw_inventory.jsonl", "parsed_documents.jsonl", "parsed_records.jsonl", "parsed_tables.jsonl", "text_chunks.jsonl"}:
                raise FileNotFoundError(source)
            continue
        originals[str(source)] = source_snapshot(source)
        destination = new.path(filename, for_write=True)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError("An existing reused input points to another source")
        elif destination.exists():
            raise ValueError("An immutable reused input was replaced with a writable file")
        else:
            destination.symlink_to(source.resolve())
        symlinks[str(destination.relative_to(selected["root"]))] = str(source.resolve())
    for filename in COPY_INPUTS:
        source = old.path(filename)
        if source.is_file():
            originals[str(source)] = source_snapshot(source)
            copy_file(source, new.path(filename, for_write=True))
    merged = []
    excluded_tables = 0
    for filename in ("evidence_candidates_deterministic.jsonl", "evidence_candidates_llm.jsonl"):
        source = old.path(filename)
        if source.is_file():
            originals[str(source)] = source_snapshot(source)
        preserved = []
        for row in rows(source):
            if row.get("recovered_table_id") or row.get("extractor_name") == "table_parameter_value_mapper":
                excluded_tables += 1
            else:
                preserved.append(row)
        atomic_rows(new.path(filename, for_write=True), preserved)
        merged.extend(preserved)
    ids = [str(row.get("evidence_id")) for row in merged]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate seed evidence IDs")
    atomic_rows(new.path("evidence_candidates.jsonl", for_write=True), merged)
    state = {**original_state, "output_dir": str(selected["raw"]), "current_stage": "deterministic_extract",
             "reprocessing_source": str(selected["original_raw"]), "stages": dict(original_state["stages"])}
    for stage in ("llm_extract", "table_recovery", "completeness", "embedding"):
        state["stages"][stage] = {"status": "not_started"}
    atomic_json(new.path("pipeline_state.json", for_write=True), state)
    originals[str(old.path("pipeline_state.json"))] = source_snapshot(old.path("pipeline_state.json"))
    # Freeze the accepted outputs too, so a completion receipt can prove that
    # no repair stage overwrote the original results or their manifests.
    for source in sorted(selected["original_final"].rglob("*")):
        if source.is_file():
            originals[str(source)] = source_snapshot(source)
    original_manifest = ROOT / "data_process/pipeline_runs" / f"{run_id}.json"
    originals[str(original_manifest)] = source_snapshot(original_manifest)
    receipt = {"accepted": True, "run_id": run_id, "created_at": now(), "originals": originals,
               "read_only_symlinks": symlinks, "preserved_seed_count": len(merged),
               "excluded_old_table_candidates": excluded_tables,
               "raw": str(selected["raw"]), "final": str(selected["final"])}
    atomic_json(selected["receipt"], receipt)
    return {"accepted": True, "preserved_seed_count": len(merged), "excluded_old_table_candidates": excluded_tables}


def verify_originals(run_id: str, *, full: bool = True) -> dict:
    selected = paths(run_id)
    receipt = read_json(selected["receipt"])
    if receipt.get("run_id") != run_id or receipt.get("accepted") is not True:
        raise ValueError("Accepted isolated input receipt is missing")
    for relative, source in receipt["read_only_symlinks"].items():
        link = selected["root"] / relative
        if not link.is_symlink() or link.resolve() != inside(source):
            raise ValueError("The read-only input binding changed")
    for source, expected in receipt["originals"].items():
        source = inside(source)
        stat = source.stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Original evidence input was modified: {source.relative_to(ROOT)}")
        if full and digest(source) != expected["sha256"]:
            raise ValueError("Original input checksum changed")
    return {"accepted": True, "checked_original_files": len(receipt["originals"]), "full_checksum": full}


def pipeline_for(run_id: str, concurrency: int):
    from pbpk_kb.pipeline_utils import PipelineConfig, UnifiedStage1Pipeline
    from pbpk_kb.processing.evidence_llm import backend_identity
    from run_evidence_pipeline import evidence_llm_preflight
    selected = paths(run_id)
    manifest = read_json(ROOT / "data_process/pipeline_runs" / f"{run_id}.json")
    model, endpoint = backend_identity()
    config = PipelineConfig(compounds=tuple(manifest["compounds"]), raw_run_root=selected["acquisition"],
                            processing_output_root=selected["root"], llm_model=model, llm_base_url=endpoint,
                            llm_concurrency=concurrency, llm_max_retries=1, llm_max_requests=None,
                            llm_preflight=evidence_llm_preflight, embedding_device="cuda",
                            embedding_batch_size=64, import_database=False, replace_database=False)
    pipeline = UnifiedStage1Pipeline(config)
    pipeline.raw_processing_dir = selected["raw"]
    pipeline.final_evidence_dir = selected["final"]
    pipeline.manifest_path = selected["root"] / "staged_pipeline_manifest.json"
    return pipeline


def enrich_candidates(run_id: str) -> dict:
    from pbpk_kb.processing.source_context import enrich_source_context
    from pbpk_kb.processing.raw_processing_paths import raw_processing_layout
    selected = paths(run_id)
    path = raw_processing_layout(selected["raw"]).path("evidence_candidates.jsonl")
    source = list(rows(path))
    cache, counts = {}, Counter()
    enriched = []
    for row in sorted(source, key=lambda value: (str(value.get("storage_path")), str(value.get("evidence_id")))):
        updated = enrich_source_context(row, project_root=ROOT, cache=cache)
        if updated.get("source_text") != row.get("source_text") or updated.get("source_text_hash") != row.get("source_text_hash"):
            raise ValueError("Context enrichment modified the source quote")
        status = ((updated.get("metadata_json") or {}).get("source_context") or {}).get("match_status", "not_supported_for_this_source_format")
        counts[status] += 1
        enriched.append(updated)
    atomic_rows(path, enriched)
    return {"accepted": True, "candidate_count": len(enriched), "context_status_counts": dict(counts)}


def final_context_guard(run_id: str) -> dict:
    from pbpk_kb.processing.source_context import enrich_source_context
    from pbpk_kb.processing.final_evidence_quality_stage import evaluate_domain_gate
    from pbpk_kb.processing.auto_quality_stage import build_kg_ready
    selected = paths(run_id)
    quality = selected["final"] / "07_llm_quality_validation"
    summary = read_json(quality / "quality_summary.json")
    if summary.get("accepted") is not True:
        raise ValueError("The final quality stage is not accepted")
    valid_path = quality / "final_auto_validated_evidence.jsonl"
    quarantine_path = quality / "final_quarantined_evidence.jsonl"
    # Preserve the pre-guard input once. This also makes an interrupted bundle
    # rewrite safely resumable without losing partition members.
    backup = selected["root"] / "context_guard_input"
    if not (backup / "receipt.json").is_file():
        for source in (valid_path, quarantine_path, quality / "quality_summary.json"):
            copy_file(source, backup / source.name)
        atomic_json(backup / "receipt.json", {"created_at": now(), "accepted": True})
    summary = read_json(backup / "quality_summary.json")
    source = list(rows(backup / valid_path.name))
    quarantine = list(rows(backup / quarantine_path.name))
    valid, review, rejected, decisions = [], [], [], []
    cache, coverage = {}, Counter()
    for row in sorted(source, key=lambda value: (str(value.get("storage_path")), str(value.get("evidence_id")))):
        updated = enrich_source_context(row, project_root=ROOT, cache=cache)
        context = ((updated.get("metadata_json") or {}).get("source_context") or {})
        status = context.get("match_status", "not_supported_for_this_source_format")
        coverage[status] += 1
        gate = evaluate_domain_gate(updated)
        if status in {"ambiguous", "unresolved"}:
            gate.update(domain_gate_status="needs_manual_review", reasons=["source_article_context_not_uniquely_resolved"])
        decisions.append({"evidence_id": row["evidence_id"], "source_context_status": status,
                          "domain_gate_status": gate["domain_gate_status"], "reasons": gate["reasons"]})
        if gate["domain_gate_status"] == "passed":
            valid.append(updated)
        else:
            updated.update(quality_status="quarantined", curation_status="needs_review",
                           domain_gate_status=gate["domain_gate_status"], domain_classification=gate["domain_classification"],
                           quality_reasons=sorted(set((row.get("quality_reasons") or []) + gate["reasons"])),
                           source_context_guard_status="review_required" if gate["domain_gate_status"] == "needs_manual_review" else "excluded")
            (review if gate["domain_gate_status"] == "needs_manual_review" else rejected).append(updated)
    quarantine += review + rejected
    original_count = len(source) + len(list(rows(backup / quarantine_path.name)))
    if len(valid) + len(quarantine) != original_count:
        raise ValueError("Source-context guard lost a partition member")
    all_ids = [str(row["evidence_id"]) for row in valid + quarantine]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Source-context partitions have duplicate evidence identities")
    nodes, edges = build_kg_ready(valid)
    for filename, values in ((valid_path.name, valid), (quarantine_path.name, quarantine),
                             ("context_review_evidence.jsonl", review), ("context_excluded_evidence.jsonl", rejected),
                             ("context_guard_decisions.jsonl", decisions), ("kg_ready_nodes.jsonl", nodes), ("kg_ready_edges.jsonl", edges)):
        atomic_rows(quality / filename, values)
    result = {"accepted": True, "retained_auto_validated_count": len(valid), "review_count": len(review),
              "excluded_count": len(rejected), "context_status_counts": dict(coverage),
              "review_ids": [row["evidence_id"] for row in review], "excluded_ids": [row["evidence_id"] for row in rejected]}
    summary.update(final_auto_validated_count=len(valid), final_quarantined_count=len(quarantine),
                   kg_node_count=len(nodes), kg_edge_count=len(edges), partition_complete=True,
                   source_context_guard=result, generated_at=now())
    atomic_json(quality / "quality_summary.json", summary)
    return result


def resumable_rag_index(pipeline) -> dict:
    """Recover only isolated, reproducible RAG outputs after interrupted writes."""
    from pbpk_kb.processing.rag_index_stage import RagIndexStage
    runner = RagIndexStage(raw_processing_dir=pipeline.raw_processing_dir,
                           final_evidence_dir=pipeline.final_evidence_dir,
                           model_name=pipeline.config.embedding_model,
                           requested_device=pipeline.config.embedding_device,
                           allow_cpu=pipeline.config.allow_cpu_embedding,
                           batch_size=pipeline.config.embedding_batch_size)
    for target in (runner.records_path, runner.embeddings_path, runner.summary_path, runner.state_path):
        inside(target, BASE)
        if target.is_symlink():
            raise ValueError("RAG outputs must be isolated writable files")
    state = read_json(runner.state_path)
    stage = (state.get("stages") or {}).get("rag_index") or {}
    summary = read_json(runner.summary_path)
    if summary.get("accepted") is True and runner.embeddings_path.is_file():
        audited = runner.audit()
        if audited.get("accepted") is not True:
            raise RuntimeError("Previously completed RAG output failed its checksum audit")
        if stage.get("accepted") is not True:
            runner._update_state_completed(state, summary)
        return {"accepted": True, "reused": True, "audit": audited}
    prepared = None
    if stage.get("status") == "prepared" and runner.records_path.is_file():
        if summary.get("records_sha256") != digest(runner.records_path):
            raise RuntimeError("Prepared RAG records changed before embedding")
        # A crash can occur after the atomic matrix write but before its receipt.
        runner.embeddings_path.unlink(missing_ok=True)
    else:
        runner.reset = True
        prepared = runner.prepare()
        runner.reset = False
    embedded = runner.embed()
    return {"accepted": embedded.get("accepted") is True, "prepare": prepared, "embed": embedded}


def verify_apply_prerequisites(run_id: str) -> dict:
    selected = paths(run_id)
    state = read_json(selected["state"])
    for stage in STAGES:
        checkpoint = (state.get("stages") or {}).get(stage) or {}
        receipt = selected["root"] / "stage_results" / f"{stage}.json"
        if (checkpoint.get("accepted") is not True or not receipt.is_file()
                or checkpoint.get("result_sha256") != digest(receipt)
                or read_json(receipt).get("accepted") is not True):
            raise RuntimeError(f"Database apply requires an intact accepted stage: {stage}")
    verify_originals(run_id, full=True)
    summary = read_json(selected["final"] / "07_llm_quality_validation/quality_summary.json")
    if summary.get("accepted") is not True or (summary.get("source_context_guard") or {}).get("accepted") is not True:
        raise RuntimeError("Final source-context isolation did not pass")
    dry = read_json(selected["root"] / "database_dry_run.json")
    if dry.get("accepted") is not True or not dry.get("input_sha256"):
        raise RuntimeError("Database dry-run has no bound input fingerprints")
    from revise_expansion_evidence import input_fingerprints
    if input_fingerprints(selected["raw"], selected["final"]) != dry["input_sha256"]:
        raise RuntimeError("Reprocessed evidence changed after the database dry-run")
    return dry


def recover_committed_revision(run_id: str, dry: dict) -> dict | None:
    """Reconcile a crash after COMMIT; never repeat a completed DB mutation."""
    import revise_expansion_evidence as revision
    import psycopg
    for audit in sorted((BASE / "db_revisions").glob("*"), reverse=True):
        before_path, after_path, transaction_path = (audit / name for name in ("before.json", "after.json", "transaction.json"))
        if not all(path.is_file() for path in (before_path, after_path, transaction_path)):
            continue
        before, transaction = read_json(before_path), read_json(transaction_path)
        if (before.get("scope", {}).get("run_id") != run_id or before.get("inputs") != dry["input_sha256"]):
            continue
        if digest(before_path) != transaction.get("before_sha256") or digest(after_path) != transaction.get("after_sha256"):
            raise RuntimeError("Database revision recovery receipt checksum changed")
        original_plan = {**before["summary"], "action": "dry-run", "database_changed": False}
        if original_plan != dry["revision_plan"]:
            continue
        after = read_json(after_path)
        with psycopg.connect(revision.imp.psycopg_dsn(revision.registered_url()), autocommit=True,
                            options="-c default_transaction_read_only=on -c statement_timeout=120000") as db:
            with db.transaction():
                if revision.snapshot(db, after["scope"], hashes=True) != after["hashes"]:
                    continue
                revision.verify_serving(db, after["scope"])
        return {**before["summary"], "accepted": True, "revision_id": transaction["revision_id"],
                "audit_dir": str(audit), "protected_rows_unchanged": True, "recovered_committed_revision": True}
    return None


def apply_validated_database(run_id: str) -> dict:
    from revise_expansion_evidence import revise, input_fingerprints
    selected = paths(run_id)
    dry = verify_apply_prerequisites(run_id)
    recovered = recover_committed_revision(run_id, dry)
    if recovered is not None:
        atomic_json(selected["root"] / "database_apply.json", recovered)
        return recovered
    fresh = revise(run_id, apply=False)
    if fresh != dry["revision_plan"]:
        raise RuntimeError("Database revision plan changed after the accepted dry-run")
    if input_fingerprints(selected["raw"], selected["final"]) != dry["input_sha256"]:
        raise RuntimeError("Evidence inputs changed while confirming the dry-run")
    result = revise(run_id, apply=True)
    if result.get("protected_rows_unchanged") is not True:
        raise RuntimeError("Applied revision lacks protected-row verification")
    result.update(accepted=True, database_modified=True)
    atomic_json(selected["root"] / "database_apply.json", result)
    return result


def verify_live_retrieval(run_id: str) -> dict:
    """Use the existing search/detail API functions with project-local GPU vectors."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
    from pbpk_kb.api.search_routes import search_evidence, semantic_search
    from pbpk_kb.api.evidence_routes import evidence_detail
    from verify_expansion_batch import (_registered_database, _local_query_vectors, _semantic_with_local_vectors,
                                        _verify_semantic_hit, _verify_source_detail, MODEL)
    selected = paths(run_id)
    apply_receipt = read_json(selected["root"] / "database_apply.json")
    if apply_receipt.get("accepted") is not True or apply_receipt.get("protected_rows_unchanged") is not True:
        raise RuntimeError("Verified database application is required before retrieval acceptance")
    final_path = selected["final"] / "07_llm_quality_validation/final_auto_validated_evidence.jsonl"
    expected = {str(row["evidence_id"]) for row in rows(final_path)}
    if not expected:
        raise RuntimeError("A batch with no accepted evidence cannot pass retrieval acceptance")
    url, config = _registered_database()
    engine = create_engine(url, connect_args={"connect_timeout": 5,
                           "options": "-c default_transaction_read_only=on -c statement_timeout=30000"})
    try:
        with Session(engine) as db:
            db.execute(text("SET TRANSACTION READ ONLY"))
            available = [dict(value) for value in db.execute(text("""SELECT evidence_id::text, evidence_type,
                attributed_compound_id::text AS compound_id, attributed_compound_name AS compound_name,
                parameter FROM auto_validated_evidence WHERE run_id=:run ORDER BY evidence_type,evidence_id"""), {"run": run_id}).mappings()]
            rag = [dict(value) for value in db.execute(text("""SELECT retrieval_record_id::text, compound_id::text,
                source_id, embedding_text, evidence_ids, embedding_model FROM rag_retrieval_record
                WHERE run_id=:run AND evidence_layer='auto_validated'"""), {"run": run_id}).mappings()]
            if {row["evidence_id"] for row in available} != expected or {str(e) for row in rag for e in row["evidence_ids"]} != expected:
                raise RuntimeError("Serving or RAG IDs differ from the isolated final accepted evidence")
            if any(row["embedding_model"] != MODEL for row in rag):
                raise RuntimeError("Database semantic index model differs from the local query encoder")
            # A new-batch scientific row is preferred to an identity-only property.
            ordered = sorted(available, key=lambda row: (row["evidence_type"] == "physicochemical", row["evidence_id"]))
            chosen = ordered[0]
            lexical = search_evidence(compound_id=chosen["compound_id"], q=chosen["parameter"], limit=500,
                                      offset=0, min_confidence=0.0, db=db)
            found = next((row for row in lexical if row["evidence_id"] in expected), None)
            if found is None:
                raise RuntimeError("Keyword search did not return the selected new-batch evidence")
            chosen = next(row for row in available if row["evidence_id"] == found["evidence_id"])
            detail = evidence_detail(chosen["evidence_type"], chosen["evidence_id"], db=db)
            source = _verify_source_detail(detail, {})
            relevant = [row for row in rag if chosen["evidence_id"] in {str(e) for e in row["evidence_ids"]}]
            query = relevant[0]["embedding_text"]
            vectors, revision = _local_query_vectors([query], config)
            semantic = _semantic_with_local_vectors(semantic_search, vectors)(q=query, source_id=relevant[0]["source_id"], limit=100, db=db)
            semantic_result = _verify_semantic_hit(semantic, {row["retrieval_record_id"] for row in relevant})
            db.rollback()
    finally:
        engine.dispose()
    result = {"accepted": True, "run_id": run_id, "checked_at": now(), "serving_count": len(expected),
              "exact_serving_and_rag_ids": True, "database_read_only": True, "ui_started": False,
              "evidence_id": chosen["evidence_id"], "compound": chosen["compound_name"],
              "keyword_search": {"passed": True, "returned_count": len(lexical)},
              "semantic_search": {"passed": True, "model": MODEL, "local_revision": revision, **semantic_result},
              "provenance": {"passed": True, "raw_path": str(source.relative_to(ROOT)), "checksum_verified": True},
              "final_evidence_sha256": digest(final_path)}
    atomic_json(selected["root"] / "retrieval_verification.json", result)
    return result


def do_stage(run_id: str, stage: str, concurrency: int, *, apply_validated: bool = False) -> dict:
    if stage == "prepare_inputs":
        return seed_inputs(run_id)
    verify_originals(run_id, full=False)
    if stage == "source_context":
        return enrich_candidates(run_id)
    if stage == "source_context_guard":
        return final_context_guard(run_id)
    if stage == "verify_originals":
        return verify_originals(run_id)
    if stage == "database_dry_run":
        from revise_expansion_evidence import revise, input_fingerprints
        selected = paths(run_id)
        frozen = input_fingerprints(selected["raw"], selected["final"])
        summary = revise(run_id, apply=False)
        if input_fingerprints(selected["raw"], selected["final"]) != frozen:
            raise RuntimeError("Evidence inputs changed during the database dry-run")
        atomic_json(paths(run_id)["root"] / "database_dry_run.json",
                    {"accepted": True, "run_id": run_id, "revision_plan": summary, "input_sha256": frozen})
        return {"accepted": True, "database_modified": False, "report": str(paths(run_id)["root"] / "database_dry_run.json")}
    if stage in {"database_apply", "verified_retrieval"}:
        if not apply_validated:
            raise RuntimeError("Database application requires explicit --apply-validated")
        return apply_validated_database(run_id) if stage == "database_apply" else verify_live_retrieval(run_id)
    pipeline = pipeline_for(run_id, concurrency)
    if stage in LLM_STAGES:
        pipeline.config.llm_preflight()
    if stage in {"classify", "llm_extract"}:
        from pbpk_kb.processing.run_llm_extraction import run_llm_extraction
        result = run_llm_extraction(processed_dir=pipeline.raw_processing_dir,
                                   llm_max_requests=0 if stage == "classify" else None,
                                   llm_model=pipeline.config.llm_model, llm_base_url=pipeline.config.llm_base_url,
                                   llm_max_retries=1, retry_backoff_seconds=1.0, concurrency=concurrency,
                                   min_chars=160, fail_fast=True, classify_only=stage == "classify",
                                   llm_include_existing=True, reuse_by_text_hash=True, reset_live_ledger=False)
        if stage == "classify":
            atomic_json(pipeline.raw_processing_dir / "00_run_manifest/llm_classification_summary.json", result)
            return {"accepted": True, "classify_only": True, "unique_requests": result.get("classify_unique_input_count"),
                    "required_chunks": result.get("llm_required_chunk_count")}
        return result
    if stage == "rag_audit":
        from pbpk_kb.processing.rag_index_stage import RagIndexStage
        return RagIndexStage(raw_processing_dir=pipeline.raw_processing_dir, final_evidence_dir=pipeline.final_evidence_dir).audit()
    if stage == "rag_index":
        return resumable_rag_index(pipeline)
    handler = getattr(pipeline, "_run_" + stage, None)
    if handler is None or stage in {"database_import", "raw_acquisition"}:
        raise ValueError("Stage is not allowed in the isolated repair workflow")
    return handler()


def process_identity(pid: int) -> dict | None:
    """Inspect only a recorded/just-spawned project PID, never enumerate processes."""
    try:
        proc = Path(f"/proc/{pid}")
        if proc.stat().st_uid != os.getuid() or (proc / "cwd").resolve() != ROOT:
            return None
        command = (proc / "cmdline").read_bytes().split(b"\0")
        script_arguments = {str(SCRIPT).encode(), str(SCRIPT.relative_to(ROOT)).encode()}
        if not script_arguments.intersection(command):
            return None
        tail = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if tail[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": tail[19]}
    except (OSError, ValueError, IndexError):
        return None


def same_process(record) -> bool:
    return bool(record and process_identity(int(record["pid"])) == record)


def terminate_worker(record) -> None:
    if not same_process(record):
        return
    os.killpg(record["pid"], signal.SIGTERM)
    until = time.monotonic() + 8
    while same_process(record) and time.monotonic() < until:
        time.sleep(.2)
    if same_process(record):
        os.killpg(record["pid"], signal.SIGKILL)


def owned_bytes(directory: Path) -> tuple[int, int]:
    total = log_max = 0
    for parent, directories, files in os.walk(directory, followlinks=False):
        directories[:] = [name for name in directories if not (Path(parent) / name).is_symlink()]
        for name in files:
            path = Path(parent) / name
            if path.is_symlink():
                continue
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            total += size
            if path.suffix == ".log":
                log_max = max(log_max, size)
    return total, log_max


def capacity_problem(total, log_max, free, *, previous, elapsed, max_bytes, max_growth, min_free):
    if total > max_bytes:
        return "isolated_output_size_limit"
    if log_max > 64 * MIB:
        return "stage_log_size_limit"
    if free < min_free:
        return "free_disk_below_safety_reserve"
    if elapsed >= 1200 and total - previous > max_growth:
        return "excessive_output_growth_in_20_minutes"
    return None


def worker_activity(record) -> int | None:
    if not same_process(record):
        return None
    try:
        tail = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return int(tail[11]) + int(tail[12])
    except (OSError, ValueError, IndexError):
        return None


def response_path(run_id: str, stage: str) -> Path | None:
    selected = paths(run_id)
    if stage == "llm_extract":
        return selected["raw"] / "05_llm_extraction/llm_extraction_responses_live.jsonl"
    if stage == "provenance_attribution":
        return selected["final"] / "01_compound_attribution/llm_attribution_responses.jsonl"
    if stage == "llm_quality_validation":
        return selected["final"] / "07_llm_quality_validation/llm_quality_results.jsonl"
    return None


def progress_signature(run_id: str, stage: str) -> tuple:
    selected = paths(run_id)
    progress = read_json(selected["raw"] / "00_run_manifest/llm_extract_progress.json") if stage in {"classify", "llm_extract"} else {}
    response = response_path(run_id, stage)
    size = response.stat().st_size if response and response.is_file() else 0
    return (size, *(int(progress.get(key) or 0) for key in ("chunks_seen", "status_rows", "api_requests_completed", "evidence_added")))


def service_failure_streak(path: Path | None, *, initial_bytes: int = 0) -> bool:
    """Only inspect new response records; never print response/error contents."""
    if path is None or not path.is_file() or path.stat().st_size <= initial_bytes:
        return False
    size = path.stat().st_size
    offset = max(initial_bytes, size - MIB)
    with path.open("rb") as stream:
        stream.seek(offset)
        lines = stream.read(MIB).splitlines()
    if offset > initial_bytes:
        lines = lines[1:]
    parsed = []
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            continue
    if len(parsed) < 3:
        return False
    markers = ("503", "504", "timeout", "timed out", "connection refused", "connection reset", "service unavailable")
    return all(any(marker in str(row.get("error") or row.get("validation") or "").lower() for marker in markers)
               for row in parsed[-3:])


def stalled_stage(stage, *, elapsed, since_progress, since_activity, timeout_seconds):
    if elapsed > timeout_seconds:
        return "explicit_stage_time_limit"
    if stage in LLM_STAGES and since_progress >= 2400 and since_activity >= 2400:
        return "llm_no_response_or_cpu_progress_for_40_minutes"
    return None


def parent_death_signal(parent_pid):
    def configure():
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != parent_pid:
            os._exit(143)
    return configure


def runtime_environment():
    from service_runner import environment
    env = environment()
    env.update(PBPK_PROJECT_BOUNDARY="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def handle_stop(signum, frame):
    global STOP
    STOP = True


def worker_main(args):
    validate_pilot(args.pilot_receipt)
    parent = read_json(BASE / "controller.pid.json")
    if not same_process(parent) or parent["pid"] != os.getppid():
        raise RuntimeError("Internal workers must belong to the active project controller")
    selected_stages = stages_for(args.apply_validated)
    if args.stage not in selected_stages:
        raise ValueError("Unknown repair stage")
    if args.stage != "prepare_inputs":
        state = read_json(paths(args.run_id[0])["state"])
        for previous in selected_stages[:selected_stages.index(args.stage)]:
            if (state.get("stages", {}).get(previous) or {}).get("accepted") is not True:
                raise ValueError(f"Required controller stage has not completed: {previous}")
    result = do_stage(args.run_id[0], args.stage, args.concurrency, apply_validated=args.apply_validated)
    if result.get("accepted") is not True:
        raise RuntimeError("The stage did not pass acceptance; preserved outputs can be resumed after diagnosis")
    atomic_json(paths(args.run_id[0])["root"] / "stage_results" / f"{args.stage}.json", result)


def run_controller(args):
    global STOP
    selected_runs = tuple(args.run_id or RUNS)
    selected_stages = stages_for(args.apply_validated)
    pilot = validate_pilot(args.pilot_receipt)
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / "controller.lock").open("a") as lock, (ROOT / "runtime/deployment/evidence_pipeline.lock").open("a") as pipeline_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(pipeline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = process_identity(os.getpid())
        if identity is None:
            raise RuntimeError("Cannot establish project controller ownership")
        atomic_json(BASE / "controller.pid.json", identity)
        signal.signal(signal.SIGTERM, handle_stop)
        signal.signal(signal.SIGINT, handle_stop)
        status = {"status": "running", "started_at": now(), "updated_at": now(), "runs": selected_runs,
                  "pid": identity, "pilot": pilot, "concurrency": args.concurrency,
                  "database_mode": "apply validated and verify retrieval" if args.apply_validated else "dry-run only",
                  "stages": selected_stages, "apply_validated": args.apply_validated, "child": None}
        initial, _ = owned_bytes(BASE / "reprocessed")
        audit_size, audit_time = initial, time.monotonic()
        atomic_json(BASE / "controller_status.json", status)
        try:
            for run_id in selected_runs:
                selected = paths(run_id)
                state = read_json(selected["state"], {"run_id": run_id, "stages": {}})
                if state.get("pilot_sha256", pilot["sha256"]) != pilot["sha256"]:
                    raise RuntimeError("A resume cannot mix stages validated by different pilot receipts")
                state["pilot_sha256"] = pilot["sha256"]
                for stage in selected_stages:
                    entry = state.setdefault("stages", {}).setdefault(stage, {})
                    if entry.get("accepted") is True:
                        continue
                    if STOP:
                        raise InterruptedError("Pause requested")
                    validate_pilot(args.pilot_receipt)
                    attempts = int(entry.get("attempts", 0))
                    if int(entry.get("failures", 0)) >= args.attempt_limit:
                        raise RuntimeError("Stage retry budget exhausted; inspect the preserved failure before extending the explicit budget")
                    total, max_log = owned_bytes(BASE / "reprocessed")
                    issue = capacity_problem(total, max_log, shutil.disk_usage(BASE).free,
                                             previous=audit_size, elapsed=time.monotonic()-audit_time,
                                             max_bytes=args.max_output_gib*GIB, max_growth=args.max_growth_gib*GIB,
                                             min_free=args.min_free_gib*GIB)
                    if issue:
                        raise RuntimeError(issue)
                    entry.update(status="running", accepted=False, started_at=now(), attempts=attempts + 1)
                    state.update(current_stage=stage, status="running", updated_at=now())
                    atomic_json(selected["state"], state)
                    logfile = selected["root"] / "logs" / f"{stage}.attempt{attempts + 1}.log"
                    logfile.parent.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, "-B", str(SCRIPT), "worker", "--execute", "--run-id", run_id,
                               "--stage", stage, "--pilot-receipt", str(inside(args.pilot_receipt)),
                               "--concurrency", str(args.concurrency)]
                    if args.apply_validated:
                        command.append("--apply-validated")
                    with logfile.open("ab") as log:
                        result_path = selected["root"] / "stage_results" / f"{stage}.json"
                        result_path.unlink(missing_ok=True)
                        child = subprocess.Popen(command, cwd=ROOT, env=runtime_environment(), stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=log, start_new_session=True,
                                                 preexec_fn=parent_death_signal(os.getpid()))
                        record = None
                        for _ in range(100):
                            record = process_identity(child.pid)
                            if record or child.poll() is not None:
                                break
                            time.sleep(.02)
                        if record is None and child.poll() is None:
                            child.terminate()
                            raise RuntimeError("Worker identity could not be established")
                        status.update(current_run=run_id, current_stage=stage, child=record, log=str(logfile), updated_at=now())
                        atomic_json(BASE / "controller_status.json", status)
                        print(f"{now()} {run_id} {stage} started", flush=True)
                        stage_started = last_progress = last_activity = time.monotonic()
                        signature = progress_signature(run_id, stage)
                        response = response_path(run_id, stage)
                        initial_response_bytes = response.stat().st_size if response and response.is_file() else 0
                        activity = worker_activity(record)
                        while child.poll() is None:
                            if STOP:
                                terminate_worker(record)
                                raise InterruptedError("Pause requested; project worker stopped")
                            total, max_log = owned_bytes(BASE / "reprocessed")
                            elapsed = time.monotonic() - audit_time
                            issue = capacity_problem(total, max_log, shutil.disk_usage(BASE).free,
                                                     previous=audit_size, elapsed=elapsed, max_bytes=args.max_output_gib*GIB,
                                                     max_growth=args.max_growth_gib*GIB, min_free=args.min_free_gib*GIB)
                            if issue:
                                terminate_worker(record)
                                raise RuntimeError(issue)
                            signature_now = progress_signature(run_id, stage)
                            activity_now = worker_activity(record)
                            stamp = time.monotonic()
                            if signature_now != signature:
                                signature, last_progress = signature_now, stamp
                            if activity_now is not None and activity_now != activity:
                                activity, last_activity = activity_now, stamp
                            issue = stalled_stage(stage, elapsed=stamp-stage_started, since_progress=stamp-last_progress,
                                                  since_activity=stamp-last_activity, timeout_seconds=args.stage_timeout_hours*3600)
                            if not issue and service_failure_streak(response, initial_bytes=initial_response_bytes):
                                issue = "three_consecutive_llm_service_failures"
                            if issue:
                                terminate_worker(record)
                                raise RuntimeError(issue)
                            if elapsed >= 1200:
                                audit_size, audit_time = total, time.monotonic()
                            status.update(updated_at=now(), isolated_output_bytes=total)
                            atomic_json(BASE / "controller_status.json", status)
                            for _ in range(20):
                                if STOP or child.poll() is not None:
                                    break
                                time.sleep(.5)
                        if child.returncode != 0:
                            entry.update(status="failed", accepted=False, failures=int(entry.get("failures", 0))+1,
                                         exit_code=child.returncode, failed_at=now(), log=str(logfile))
                            atomic_json(selected["state"], state)
                            raise RuntimeError(f"Stage {stage} failed; no automatic restart loop")
                    result_path = selected["root"] / "stage_results" / f"{stage}.json"
                    result = read_json(result_path)
                    if result.get("accepted") is not True:
                        raise RuntimeError("Worker exited without an accepted stage receipt")
                    entry.update(status="completed", accepted=True, completed_at=now(), result_path=str(result_path), result_sha256=digest(result_path))
                    state["updated_at"] = now()
                    atomic_json(selected["state"], state)
                    status["child"] = None
                    print(f"{now()} {run_id} {stage} accepted", flush=True)
                state.update(status="completed", current_stage=selected_stages[-1], completed_at=now())
                atomic_json(selected["state"], state)
            status.update(status="completed_and_verified" if args.apply_validated else "completed_dry_run",
                          completed_at=now(), child=None)
        except BaseException as exc:
            if same_process(status.get("child")):
                terminate_worker(status["child"])
            status.update(status="paused" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed",
                          error_type=type(exc).__name__, error=str(exc), child=None)
            if "entry" in locals() and entry.get("status") == "running":
                entry.update(status=status["status"], accepted=False, stopped_at=now())
                if status["status"] == "failed":
                    entry["failures"] = int(entry.get("failures", 0)) + 1
                state.update(status=status["status"], updated_at=now())
                atomic_json(selected["state"], state)
            raise
        finally:
            status["updated_at"] = now()
            atomic_json(BASE / "controller_status.json", status)


def latest_json_rows(path: Path, *, max_bytes: int = 256 * 1024) -> list[dict]:
    """Bound status reads independently of the lifetime of the extraction log."""
    if not path.is_file():
        return []
    size = path.stat().st_size
    offset = max(0, size - max_bytes)
    with path.open("rb") as stream:
        stream.seek(offset)
        lines = stream.read(max_bytes).splitlines()
    if offset:
        lines = lines[1:]
    result = []
    for line in lines:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except (ValueError, UnicodeDecodeError):
            continue
    return result


def display_llm_phase(state: dict, progress: dict) -> dict:
    """Correct a stale first-response label without changing scientific state."""
    result = dict(progress)
    if (state.get("current_stage") != "llm_extract" or not state.get("worker_alive")
            or progress.get("status") != "building_high_precision_routing_index"
            or state.get("current_run") not in RUNS):
        return result
    selected = paths(state["current_run"])
    entry = read_json(selected["state"]).get("stages", {}).get("llm_extract") or {}
    prompt = (state.get("pilot") or {}).get("prompt")
    try:
        began = datetime.fromisoformat(entry["started_at"])
    except (KeyError, ValueError):
        return result
    submitted = []
    for request in latest_json_rows(selected["raw"] / "05_llm_extraction/llm_extraction_requests_live.jsonl"):
        try:
            belongs = (request.get("prompt_template_version") == prompt and prompt
                       and datetime.fromisoformat(request["created_at"]) >= began)
        except (KeyError, ValueError, TypeError):
            belongs = False
        if belongs:
            submitted.append(request)
    backend = read_json(ROOT / "runtime/deployment/evidence_expansion_20261009/backend/status.json")
    if submitted and backend.get("state") == "generating":
        result.update(status="waiting_for_llm_response", recorded_phase=progress["status"],
                      requests_observed_in_tail=len(submitted))
    return result


def progress_status() -> dict:
    state = read_json(BASE / "controller_status.json")
    selected_stages = state.get("stages") or STAGES
    complete = sum((read_json(paths(run_id)["state"]).get("stages", {}).get(stage) or {}).get("accepted") is True
                   for run_id in (state.get("runs") or RUNS) for stage in selected_stages)
    total = len(state.get("runs") or RUNS) * len(selected_stages)
    state.update(completed_stages=complete, total_stages=total, controller_alive=same_process(state.get("pid")),
                 worker_alive=same_process(state.get("child")))
    if state.get("current_run") in RUNS:
        recovery_dir = BASE / "automatic_bounded_recovery" / state["current_run"]
        if not (recovery_dir / "status.json").is_file():
            recovery_dir = BASE / "bounded_recovery" / state["current_run"]
        recovery = read_json(recovery_dir / "status.json")
        if recovery:
            identity = read_json(recovery_dir / "worker.pid.json")
            alive = False
            try:
                proc = Path("/proc") / str(int(identity["pid"]))
                if proc.stat().st_uid == os.getuid() and (proc / "cwd").resolve() == ROOT:
                    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                    alive = (fields[0] != "Z" and fields[19] == identity["start_ticks"]
                             and str(ROOT / "scripts/deployment/recover_bounded_evidence.py").encode()
                             in (proc / "cmdline").read_bytes().split(b"\0"))
            except (OSError, ValueError, KeyError, IndexError):
                pass
            state["bounded_recovery"] = {**recovery, "worker_alive": alive}
    followup = read_json(BASE / "followup/status.json")
    if followup:
        identity = followup.get("guardian") or {}
        alive = False
        try:
            proc = Path("/proc") / str(int(identity["pid"]))
            if proc.stat().st_uid == os.getuid() and (proc / "cwd").resolve() == ROOT:
                fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                alive = (fields[0] not in {"Z", "T", "t"} and fields[19] == identity["start_ticks"]
                         and str(ROOT / "scripts/deployment/follow_evidence_expansion.py").encode()
                         in (proc / "cmdline").read_bytes().split(b"\0"))
        except (OSError, ValueError, KeyError, IndexError):
            pass
        state["quality_followup"] = {key: followup.get(key) for key in
                                     ("status", "reason", "checked_at", "last_audit_at", "recovery_attempts")}
        state["quality_followup"]["guardian_alive"] = alive
    if state.get("current_run") in RUNS and state.get("current_stage") in {"classify", "llm_extract"}:
        selected = paths(state["current_run"])
        progress = read_json(selected["raw"] / "00_run_manifest/llm_extract_progress.json")
        progress = display_llm_phase(state, progress)
        classification = read_json(selected["raw"] / "00_run_manifest/llm_classification_summary.json")
        state["llm_progress"] = {"phase": progress.get("status"), "counts": {key: progress.get(key, 0) for key in
                                 ("chunks_seen", "status_rows", "api_requests_submitted", "api_requests_completed", "evidence_added", "pending")},
                                 "expected_unique_requests": classification.get("classify_unique_input_count"),
                                 "requests_observed_in_tail": progress.get("requests_observed_in_tail"),
                                 "updated_at": progress.get("updated_at")}
    elif state.get("current_run") in RUNS and state.get("current_stage") == "provenance_attribution":
        directory = paths(state["current_run"])["final"] / "01_compound_attribution"
        responses, requests = directory / "llm_attribution_responses.jsonl", directory / "llm_attribution_requests.jsonl"
        if responses.exists() and requests.exists() and responses.stat().st_size < 32*MIB and requests.stat().st_size < 32*MIB:
            values = {}
            with responses.open() as stream:
                for line in stream:
                    try:
                        value = json.loads(line); values[value["evidence_id"]] = value
                    except (ValueError, KeyError):
                        continue  # A concurrently written trailing line is not a completed response.
            with requests.open() as stream:
                count = sum(bool(line.strip()) for line in stream)
            state["attribution_progress"] = {"responded": len(values), "submitted": count,
                "valid": sum((v.get("validation") or {}).get("status") == "valid" for v in values.values())}
    legacy_dir = BASE / "legacy_curation_20261022/context_followup"
    legacy = read_json(legacy_dir / "status.json")
    if legacy:
        alive = False
        try:
            proc = Path("/proc") / (legacy_dir / "worker.pid").read_text().strip()
            alive = (proc.stat().st_uid == os.getuid() and (proc / "cwd").resolve() == ROOT
                     and b"review_legacy_context.py" in (proc / "cmdline").read_bytes()
                     and (proc / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z")
        except (OSError, ValueError, IndexError):
            pass
        state["legacy_context_review"] = {**legacy, "worker_alive": alive}
    return state


def print_status(as_json=False):
    status = progress_status()
    if as_json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    done, total = status["completed_stages"], status["total_stages"]
    width = 34
    filled = int(width * done / max(total, 1))
    print(f"PBPK evidence repair  [{'━' * filled}{'─' * (width-filled)}] {done}/{total} stages ({done/max(total,1):.1%})")
    print(f"{status.get('status', 'not_started')} | {status.get('current_run', '—')} | {status.get('current_stage', '—')}")
    print(f"controller={'running' if status['controller_alive'] else 'stopped'} worker={'running' if status['worker_alive'] else 'stopped'} | database: {status.get('database_mode', 'dry-run only')}")
    if status.get("error"):
        print("Last controller stop:", status["error"])
    if status.get("quality_followup"):
        follow = status["quality_followup"]
        print(f"Follow-up: {follow.get('status')} | guardian={'running' if follow['guardian_alive'] else 'stopped'} | "
              f"{follow.get('reason')} | automatic recoveries={sum((follow.get('recovery_attempts') or {}).values())}")
    if status.get("bounded_recovery"):
        recovery = status["bounded_recovery"]
        print(f"Targeted recovery: {recovery.get('done', 0)}/{recovery.get('total', 0)} decisions | "
              f"{recovery.get('parent_requests', 0)} failed requests | {recovery.get('status')} | "
              f"worker={'running' if recovery['worker_alive'] else 'stopped'} | "
              f"unresolved={recovery.get('failed', 0)}")
    if status.get("llm_progress"):
        progress = status["llm_progress"]
        print("LLM:", progress.get("phase"), "expected unique requests:", progress.get("expected_unique_requests"), "counts:", json.dumps(progress.get("counts"), ensure_ascii=False))
        if progress.get("requests_observed_in_tail"):
            print("Requests submitted in the current worker (bounded log tail):", progress["requests_observed_in_tail"])
    if status.get("attribution_progress"):
        progress = status["attribution_progress"]
        print(f"Source attribution: {progress['responded']} responses / {progress['submitted']} submitted so far; {progress['valid']} passed")
    if status.get("legacy_context_review"):
        legacy = status["legacy_context_review"]
        print(f"Old evidence context review: {legacy.get('done', 0)}/{legacy.get('total', 0)} | {legacy.get('status')} | worker={'running' if legacy['worker_alive'] else 'stopped'}")
        print("Review outcomes:", json.dumps(legacy.get("counts", {}), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "status", "watch", "start", "run", "worker", "pause"), default="status", nargs="?")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--apply-validated", action="store_true", help="Apply only after accepted stages, an unchanged dry-run, and source-context isolation; then test retrieval")
    parser.add_argument("--run-id", choices=RUNS, action="append")
    parser.add_argument("--pilot-receipt", type=Path)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--stage", choices=APPLY_STAGES)
    parser.add_argument("--attempt-limit", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-output-gib", type=int, default=35)
    parser.add_argument("--max-growth-gib", type=int, default=5)
    parser.add_argument("--min-free-gib", type=int, default=12)
    parser.add_argument("--stage-timeout-hours", type=int, choices=range(1, 169), default=48)
    parser.add_argument("--interval", type=int, choices=range(2, 61), default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if min(args.max_output_gib, args.max_growth_gib, args.min_free_gib) <= 0:
        parser.error("Disk limits and the reserve must be positive")
    if args.action == "plan":
        print(json.dumps({"run_ids": args.run_id or RUNS, "stages": stages_for(args.apply_validated), "output_root": str(BASE / "reprocessed"),
                          "starts_work": False, "database_apply": args.apply_validated, "requires_passed_pilot_v3_or_later": True,
                          "read_only_input_files": READ_ONLY_INPUTS, "request_concurrency": args.concurrency,
                          "old_30_compounds_untouched": True}, ensure_ascii=False, indent=2))
        return
    if args.action == "status":
        print_status(args.json)
        return
    if args.action == "watch":
        while True:
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print_status(args.json)
            time.sleep(args.interval)
    if not args.execute:
        parser.error("Mutating actions require explicit --execute")
    if args.action == "pause":
        if (BASE / "followup").is_dir():
            atomic_json(BASE / "followup/manual_pause", {"requested_at": now(), "reason": "explicit_quality_controller_pause"})
        record = read_json(BASE / "controller.pid.json")
        if same_process(record):
            os.kill(record["pid"], signal.SIGTERM)
            print("Pause requested for this project repair controller")
        else:
            print("No matching project repair controller is running")
        return
    if args.pilot_receipt is None:
        parser.error("--pilot-receipt is required before execution")
    validate_pilot(args.pilot_receipt)
    if args.action == "worker":
        if not args.stage or not args.run_id or len(args.run_id) != 1:
            parser.error("worker requires exactly one --run-id and --stage")
        worker_main(args)
    elif args.action == "run":
        run_controller(args)
    elif args.action == "start":
        BASE.mkdir(parents=True, exist_ok=True)
        # This lock only serializes launch attempts. The detached controller
        # owns the persistent execution lock for its full lifetime.
        with (BASE / "launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if same_process(read_json(BASE / "controller.pid.json")):
                raise RuntimeError("The isolated repair controller is already running")
            command = [sys.executable, "-B", str(SCRIPT), "run", "--execute", "--pilot-receipt", str(inside(args.pilot_receipt)),
                       "--concurrency", str(args.concurrency), "--attempt-limit", str(args.attempt_limit),
                       "--max-output-gib", str(args.max_output_gib), "--max-growth-gib", str(args.max_growth_gib),
                       "--min-free-gib", str(args.min_free_gib)]
            command += ["--stage-timeout-hours", str(args.stage_timeout_hours)]
            if args.apply_validated:
                command.append("--apply-validated")
            for run_id in args.run_id or []:
                command += ["--run-id", run_id]
            logfile = BASE / "controller.log"
            if logfile.exists() and logfile.stat().st_size > 64 * MIB:
                raise RuntimeError("Controller log exceeded the limit; inspect it before another launch")
            with logfile.open("ab") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=runtime_environment(), stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=log, start_new_session=True)
            for _ in range(100):
                identity = process_identity(process.pid)
                if (identity and read_json(BASE / "controller.pid.json") == identity) or process.poll() is not None:
                    break
                time.sleep(.02)
            if process.poll() is not None:
                raise RuntimeError("Controller exited during launch; inspect its project log")
            print(json.dumps({"controller_pid": process.pid, "log": str(logfile), "database_apply": args.apply_validated}))


if __name__ == "__main__":
    main()
