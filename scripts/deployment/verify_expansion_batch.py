"""Read-only serving acceptance for one completed knowledge-expansion batch.

Use the existing search/detail implementations and locally cached PubMedBERT on
GPU. No HTTP model endpoint, API server, UI, database mutation or model download.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
from types import FunctionType

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_process/knowledge_expansion_20261009"
MODEL = "NeuML/pubmedbert-base-embeddings"


def _read(path):
    return json.loads(path.read_text())


def _inside(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError("Acceptance input escapes this project")
    return path


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _registered_database():
    from dotenv import dotenv_values
    from sqlalchemy.engine import make_url

    pidfile = ROOT / "runtime/postgres-data/postmaster.pid"
    lines = pidfile.read_text().splitlines()
    proc = Path(f"/proc/{int(lines[0])}")
    if (pidfile.stat().st_uid != os.getuid() or proc.stat().st_uid != os.getuid()
            or Path(lines[1]).resolve() != ROOT / "runtime/postgres-data"
            or (proc / "cwd").resolve() != ROOT / "runtime/postgres-data"):
        raise RuntimeError("Project database ownership could not be verified")
    config = dotenv_values(ROOT / "runtime/deployment/app.env", interpolate=False)
    url = make_url(config["DATABASE_URL"])
    ports = _read(ROOT / "runtime/deployment/ports.json")
    if (url.host != "127.0.0.1" or url.port != int(lines[3])
            or url.port != ports["postgres"] or url.database != "pbpk_kb"):
        raise RuntimeError("Unexpected project database address")
    return url, config


def _check_final_ids(expected, available, linked):
    missing = expected - available
    unlinked = expected - linked
    if missing or unlinked:
        raise RuntimeError(f"Accepted evidence missing from serving/index: {len(missing)}/{len(unlinked)}")
    return {"expected": len(expected), "available": len(expected & available),
            "missing": 0, "missing_rag_evidence_links": 0}


def _target_rows(rows, target):
    existing = target.get("existing_compound_id")
    names = {str(value).casefold() for value in [target["compound_name"], target.get("search_name"),
                                               *target.get("aliases", [])] if value}
    if existing:
        return [r for r in rows if str(r.get("attributed_compound_id") or r.get("compound_id")) == existing]
    return [r for r in rows if str(r.get("attributed_compound_name") or r.get("compound_name") or "").casefold() in names]


def _batch_targets(compounds, plan):
    """Resolve a frozen ordered batch under its canonical or search spelling."""
    targets = {row["compound_name"]: row for row in plan["targets"]}
    for batch in plan["batches"]:
        canonical = batch["compound_names"]
        selected = [targets[name] for name in canonical]
        search = [row.get("search_name") or row["compound_name"] for row in selected]
        if len(set(compounds)) == len(compounds) and compounds in (canonical, search):
            return selected
    raise RuntimeError("Batch compounds do not match the registered expansion targets")


def _import_identity_view(rows, imported, mapping):
    """Require the importer to have used precisely the independently rebuilt map."""
    if imported.get("identity_mapping", {"version": 1, "mappings": []}) != mapping.receipt:
        raise RuntimeError("Imported identity mapping differs from frozen/source-verified bindings")
    return [mapping.record(row) for row in rows]


def _local_query_vectors(texts, config):
    """Load only a project-owned, already cached model; never use the old server."""
    import torch
    from sentence_transformers import SentenceTransformer

    if config.get("PBPK_EMBEDDING_MODEL", MODEL) != MODEL:
        raise RuntimeError("Serving embedding model differs from the frozen RAG model")
    cache = _inside(config.get("HF_HUB_CACHE") or ROOT / "runtime/hf_cache/embedding")
    repository = cache / "models--NeuML--pubmedbert-base-embeddings"
    revision = (repository / "refs/main").read_text().strip()
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise RuntimeError("Local embedding revision is invalid")
    snapshot = _inside(repository / "snapshots" / revision)
    if not snapshot.is_dir():
        raise RuntimeError("Project embedding snapshot is missing")
    # Check symlinked HF blobs too, before the model loader reads anything.
    for file in snapshot.rglob("*"):
        _inside(file)
    device = config.get("PBPK_EMBEDDING_DEVICE", "cuda:0")
    if not re.fullmatch(r"cuda(?::\d+)?", device) or not torch.cuda.is_available():
        raise RuntimeError("Project GPU is unavailable for acceptance embeddings")
    if torch.cuda.mem_get_info(device)[0] < 1536 * 1024**2:
        raise RuntimeError("Insufficient free GPU memory for local embedding acceptance")
    model = SentenceTransformer(str(snapshot), device=device, local_files_only=True,
                                trust_remote_code=False)
    try:
        vectors = model.encode(texts, batch_size=min(16, len(texts)),
                               normalize_embeddings=True, show_progress_bar=False)
        from pbpk_kb.processing.rag_index_stage import validate_embedding_matrix
        import numpy as np
        validate_embedding_matrix(vectors, len(texts), 768, np)
        return {t: v.tolist() for t, v in zip(texts, vectors)}, revision
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _semantic_with_local_vectors(search, vectors):
    # Clone globals instead of monkeypatching a process-global API function.
    local = FunctionType(search.__code__, {**search.__globals__, "embed_query": vectors.__getitem__},
                         search.__name__, search.__defaults__, search.__closure__)
    local.__kwdefaults__ = search.__kwdefaults__
    return local


def _verify_semantic_hit(hits, eligible_ids):
    matches = [r for r in hits if r.get("retrieval_record_id") in eligible_ids]
    if not matches:
        raise RuntimeError("Semantic query returned no current-run record for the target compound")
    return {"returned_count": len(hits), "current_run_matches": len(matches),
            "matched_retrieval_record_id": matches[0]["retrieval_record_id"],
            "score": float(matches[0]["score"])}


def _verify_source_detail(detail, checksums):
    fields, document = detail["extracted_fields"], detail["source_document"]
    if not document or not fields.get("source_location") or not fields.get("checksum"):
        raise RuntimeError("Evidence provenance is incomplete: " + str(fields.get("evidence_id")))
    source = _inside(fields["storage_path"])
    if (not document.get("raw_storage_path") or not document.get("raw_checksum")
            or _inside(document["raw_storage_path"]) != source
            or document["raw_checksum"] != fields["checksum"]):
        raise RuntimeError("Evidence and source registry provenance disagree")
    if source not in checksums:
        checksums[source] = _sha(source)
    if checksums[source] != fields["checksum"]:
        raise RuntimeError("Raw source checksum differs: " + str(fields.get("evidence_id")))
    return source


def verify_batch(run_id: str) -> dict:
    if not re.fullmatch(r"run_\d{8}T\d{6}Z", run_id):
        raise ValueError("Expected a project evidence run identifier")
    manifest = _read(ROOT / f"data_process/pipeline_runs/{run_id}.json")
    if manifest.get("run_id") != run_id or manifest.get("status") != "completed":
        raise RuntimeError("Batch pipeline is not complete")
    compounds = manifest["compounds"]
    targets = _read(DATA / "targets.json")
    batch_targets = _batch_targets(compounds, targets)
    final = _inside(manifest["final_evidence_dir"])
    raw = _inside(manifest["raw_processing_dir"])
    if final != ROOT / f"data_process/final_evidence/{run_id}" or raw != ROOT / f"data_process/raw_processing/{run_id}":
        raise RuntimeError("Unexpected batch input paths")
    quality = _read(final / "07_llm_quality_validation/quality_summary.json")
    imported = _read(final / "00_run_manifest/database_import_summary.json")
    if quality.get("accepted") is not True or imported.get("accepted") is not True or imported.get("replace") is not False:
        raise RuntimeError("Batch quality/import was not accepted as append-only")
    evidence_path = final / "07_llm_quality_validation/final_auto_validated_evidence.jsonl"
    final_rows = [json.loads(line) for line in evidence_path.read_text().splitlines() if line]
    url, config = _registered_database()
    from pbpk_kb.storage.expansion_identity_mapping import (
        load_expansion_identity_mapping, lookup_registered_identities,
    )
    identity_mapping = load_expansion_identity_mapping(
        final, raw, lambda ids: lookup_registered_identities(url, ids), root=ROOT,
    )
    final_rows = _import_identity_view(final_rows, imported, identity_mapping)
    expected = {str(r["evidence_id"]) for r in final_rows}
    if not expected or any(r["run_id"] != run_id or r.get("quality_status") != "auto_validated" for r in final_rows):
        raise RuntimeError("Final evidence is empty, from another run, or unvalidated")
    if imported.get("serving_evidence_count") != len(expected) or quality.get("final_auto_validated_count") != len(expected):
        raise RuntimeError("Final accepted evidence count disagrees with the import summary")
    from pbpk_kb.processing.rag_index_stage import RagIndexStage
    rag_audit = RagIndexStage(raw_processing_dir=raw, final_evidence_dir=final).audit()
    if not rag_audit["accepted"] or rag_audit["embedding_model"] != MODEL:
        raise RuntimeError("Batch RAG file audit failed")
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
    from pbpk_kb.api.search_routes import search_evidence, semantic_search
    from pbpk_kb.api.evidence_routes import evidence_detail
    engine = create_engine(url, connect_args={"connect_timeout": 5,
        "options": "-c default_transaction_read_only=on -c statement_timeout=30000"})
    try:
        with Session(engine) as db:
            db.execute(text("SET TRANSACTION READ ONLY"))
            available = set(db.execute(text("SELECT evidence_id::text FROM auto_validated_evidence WHERE evidence_id::text = ANY(:ids)"), {"ids": sorted(expected)}).scalars())
            rag = [dict(r) for r in db.execute(text("""SELECT retrieval_record_id::text, compound_id::text,
                compound_name, source_id, embedding_text, evidence_ids, embedding_model
                FROM rag_retrieval_record WHERE run_id=:run AND evidence_layer='auto_validated'"""), {"run": run_id}).mappings()]
            if any(r["embedding_model"] != MODEL for r in rag):
                raise RuntimeError("Database RAG model differs from the local query encoder")
            links = {str(e) for r in rag for e in r["evidence_ids"]}
            id_check = _check_final_ids(expected, available, links)
            # Check every imported source, including API fields and table cells.
            details, checksums = {}, {}
            provenance_kinds = Counter()
            for row in final_rows:
                eid = str(row["evidence_id"])
                detail = evidence_detail(row["evidence_type"], eid, db=db)
                _verify_source_detail(detail, checksums)
                details[eid] = detail
                provenance_kinds[detail["source_document"].get("source_kind", "parsed_document")] += 1
            per_compound, queries, eligible = [], [], []
            for compound, target in zip(compounds, batch_targets):
                candidates = _target_rows(final_rows, target)
                if not candidates:
                    raise RuntimeError(f"No accepted evidence for target compound: {compound}")
                compound_id = str(candidates[0].get("attributed_compound_id") or candidates[0]["compound_id"])
                current_ids = {str(r["evidence_id"]) for r in candidates}
                rows = search_evidence(compound_id=compound_id, limit=500, offset=0, min_confidence=0.0, db=db)
                example = next((r for r in rows if r["evidence_id"] in current_ids), None)
                if not example:
                    raise RuntimeError(f"No current-run structured search hit: {compound}")
                keyword = example["parameter_name"]
                hits = search_evidence(compound_id=compound_id, q=keyword, limit=500, offset=0, min_confidence=0.0, db=db)
                if example["evidence_id"] not in {r["evidence_id"] for r in hits}:
                    raise RuntimeError(f"Keyword search missed accepted evidence: {compound}")
                detail = details[example["evidence_id"]]
                fields, document = detail["extracted_fields"], detail["source_document"]
                source = _verify_source_detail(detail, checksums)
                relevant = [r for r in rag if r["compound_id"] == compound_id and set(r["evidence_ids"]) & current_ids]
                if not relevant:
                    raise RuntimeError(f"No current-run semantic record: {compound}")
                selected = next((r for r in relevant if example["evidence_id"] in r["evidence_ids"]), relevant[0])
                queries.append(selected["embedding_text"])
                eligible.append({r["retrieval_record_id"] for r in relevant})
                per_compound.append({"compound": compound, "compound_id": compound_id,
                    "accepted_evidence_count": len(current_ids), "evidence_types": dict(Counter(r["evidence_type"] for r in candidates)),
                    "lexical_search": {"passed": True, "keyword": keyword, "returned_count": len(hits), "matched_evidence_id": example["evidence_id"]},
                    "provenance": {"passed": True, "source_document_present": True, "source_id": fields["source_id"],
                        "raw_sha256_verified": fields["checksum"], "raw_path": str(source.relative_to(ROOT))},
                    "semantic_source_id": selected["source_id"]})
            vectors, revision = _local_query_vectors(queries, config)
            local_search = _semantic_with_local_vectors(semantic_search, vectors)
            for row, query, ids in zip(per_compound, queries, eligible):
                hits = local_search(q=query, source_id=row.pop("semantic_source_id"), limit=100, db=db)
                row["semantic_search"] = {"passed": True, **_verify_semantic_hit(hits, ids)}
            db.rollback()
    finally:
        engine.dispose()
    return {"run_id": run_id, "checked_at": datetime.now(timezone.utc).isoformat(), "accepted": True, "passed": True,
        "database_read_only": True, "ui_started": False, "external_embedding_service_used": False,
        "semantic_encoder": {"model": MODEL, "local_revision": revision, "device": "cuda", "queries": len(queries)},
        "final_evidence_sha256": _sha(evidence_path), "serving_ids": id_check,
        "identity_mapping": identity_mapping.receipt,
        "all_evidence_provenance": {"checked": len(details), "raw_files_checked": len(checksums),
                                    "source_kinds": dict(provenance_kinds), "passed": True},
        "rag_file_audit": rag_audit, "compound_checks": per_compound,
        "scope": "Serving completeness and source-linked retrieval, not clinical validation or expert assessment."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(verify_batch(args.run_id), ensure_ascii=False, indent=2))
