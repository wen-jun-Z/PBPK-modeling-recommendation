"""Apply source-verified table omissions without overwriting the completed run.

The ordinary importer, transaction scope, protected-row checks and RAG checks
are reused. This repairs a completed batch; running batches use the fixed gate.
"""
import argparse
from collections import Counter
from pathlib import Path
import json

import repair_expansion_quality as ctl
import revise_expansion_evidence as revision
from pbpk_kb.processing.auto_quality_stage import deterministic_repair, evaluate_evidence
from pbpk_kb.processing.final_evidence_quality_stage import evaluate_domain_gate

OUT = ctl.BASE / "table_cell_repair_20261026" / ctl.RUNS[0]


def prepare():
    source = ctl.paths(ctl.RUNS[0])["final"]
    final = OUT / "final"
    if (OUT / "prepared.json").exists():
        return ctl.read_json(OUT / "prepared.json")
    state = ctl.read_json(ctl.paths(ctl.RUNS[0])["state"])
    if not state["stages"]["verified_retrieval"].get("accepted"):
        raise RuntimeError("Only a completed, verified batch may receive this followup")
    accepted_path = source / "07_llm_quality_validation/final_auto_validated_evidence.jsonl"
    quarantine_path = source / "07_llm_quality_validation/final_quarantined_evidence.jsonl"
    accepted, quarantine = list(ctl.rows(accepted_path)), list(ctl.rows(quarantine_path))
    promoted, remaining, decisions = [], [], []
    for row in quarantine:
        if row.get("extraction_method") != "source_structured_table":
            remaining.append(row)
            continue
        repaired, repairs = deterministic_repair(row)
        decision = evaluate_evidence(repaired, repairs)
        domain = evaluate_domain_gate(repaired)
        proof = decision.get("table_source_validation") or {}
        good = (proof.get("accepted") is True and decision["quality_status"] == "auto_validated"
                and domain["domain_gate_status"] == "passed" and not row.get("review_blockers"))
        decisions.append({"evidence_id": row["evidence_id"], "promoted": good, "decision": decision, "domain": domain})
        if good:
            promoted.append({**repaired, **decision, "deterministic_repairs": repairs,
                             "metadata_json": {**repaired.get("metadata_json", {}),
                                               "table_cell_repair": {"receipt": str(OUT / "decisions.json"), "source_verified": True}}})
        else:
            remaining.append(row)
    if not promoted:
        raise RuntimeError("No table evidence passed the independent source check")
    # These rows had already passed attribution, domain, both deduplication
    # stages, and conflict analysis. No previously excluded member is revived.
    reviewable = {r["evidence_id"] for r in ctl.rows(source / "05_evidence_layers/reviewable_evidence.jsonl")}
    if not {r["evidence_id"] for r in promoted} <= reviewable:
        raise RuntimeError("Proposed table evidence skipped earlier scientific gates")
    all_ids = [r["evidence_id"] for r in accepted + promoted]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Duplicate serving evidence IDs")
    final.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == "07_llm_quality_validation":
            continue
        target = final / item.name
        if not target.exists():
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    quality = final / "07_llm_quality_validation"
    quality.mkdir(exist_ok=True)
    ctl.atomic_rows(quality / "final_auto_validated_evidence.jsonl", accepted + promoted)
    ctl.atomic_rows(quality / "final_quarantined_evidence.jsonl", remaining)
    summary = {**ctl.read_json(source / "07_llm_quality_validation/quality_summary.json"),
               "generated_at": ctl.now(), "final_auto_validated_count": len(all_ids),
               "final_quarantined_count": len(remaining), "table_cell_followup_promoted": len(promoted),
               "source_quality_summary_sha256": ctl.digest(source / "07_llm_quality_validation/quality_summary.json")}
    ctl.atomic_json(quality / "quality_summary.json", summary)
    ctl.atomic_json(OUT / "decisions.json", decisions)
    report = {"run_id": ctl.RUNS[0], "retained": len(accepted), "promoted": len(promoted), "serving_total": len(all_ids),
              "promoted_ids": [r["evidence_id"] for r in promoted], "remaining_quarantined": len(remaining),
              "per_compound": dict(Counter(r["compound_name"] for r in promoted)),
              "original_accepted_sha256": ctl.digest(accepted_path), "original_quarantine_sha256": ctl.digest(quarantine_path),
              "code_sha256": {name: ctl.digest(ctl.ROOT / name) for name in [
                  "src/pbpk_kb/processing/table_evidence_validation.py", "src/pbpk_kb/processing/auto_quality_stage.py",
                  "src/pbpk_kb/parsers/table_structure.py", "src/pbpk_kb/processing/final_evidence_quality_stage.py"]}}
    ctl.atomic_json(OUT / "prepared.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    report = prepare()
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if args.verify:
        verify(report)
        return
    if not args.apply:
        return
    if (OUT / "committed.json").exists():
        print("This followup was already committed; no second write.")
        return
    for name, expected in report["code_sha256"].items():
        if ctl.digest(ctl.ROOT / name) != expected:
            raise RuntimeError("Validated table code changed")
    final = OUT / "final"
    dry = revision.revise(ctl.RUNS[0], final_dir=final)
    ctl.atomic_json(OUT / "database_dry_run.json", dry)
    applied = revision.revise(ctl.RUNS[0], final_dir=final, apply=True)
    ctl.atomic_json(OUT / "committed.json", applied)
    print(json.dumps(applied, ensure_ascii=False), flush=True)


def verify(report):
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
    from verify_expansion_batch import (_registered_database, _local_query_vectors, _semantic_with_local_vectors,
                                        _verify_source_detail, _verify_semantic_hit)
    from pbpk_kb.api.search_routes import search_evidence, semantic_search
    from pbpk_kb.api.evidence_routes import evidence_detail
    url, config = _registered_database()
    engine = create_engine(url, connect_args={"options": "-c default_transaction_read_only=on -c statement_timeout=30000"})
    checksums, per_compound, keyword_checked = {}, {}, 0
    with Session(engine) as db:
        rows = list(db.execute(text("SELECT to_jsonb(t) FROM auto_validated_evidence t WHERE evidence_id::text=ANY(:ids)"),
                               {"ids": report["promoted_ids"]}).scalars())
        if {r["evidence_id"] for r in rows} != set(report["promoted_ids"]):
            raise RuntimeError("Promoted table records are not all visible")
        rag = list(db.execute(text("SELECT retrieval_record_id::text,compound_name,evidence_ids,embedding_text,source_id FROM rag_retrieval_record WHERE evidence_ids ?| :ids"),
                              {"ids": report["promoted_ids"]}).mappings())
        keyword_cache = {}
        for row in rows:
            _verify_source_detail(evidence_detail(row["evidence_type"], row["evidence_id"], db=db), checksums)
            key = (row["attributed_compound_id"], row["parameter"])
            if key not in keyword_cache:
                keyword_cache[key] = {h["evidence_id"] for h in search_evidence(compound_id=key[0], q=key[1], limit=500, offset=0, min_confidence=0, db=db)}
            if row["evidence_id"] not in keyword_cache[key]:
                raise RuntimeError("Keyword search missed repaired evidence: " + row["evidence_id"])
            keyword_checked += 1
            per_compound.setdefault(row["attributed_compound_name"], next(r for r in rag if row["evidence_id"] in r["evidence_ids"]))
        queries = [r["embedding_text"] for r in per_compound.values()]
        vectors, embedding_revision = _local_query_vectors(queries, config)
        search = _semantic_with_local_vectors(semantic_search, vectors)
        semantic = {drug: _verify_semantic_hit(search(q=row["embedding_text"], source_id=row["source_id"], limit=100, db=db),
                                               {row["retrieval_record_id"]}) for drug, row in per_compound.items()}
        counts = [dict(r) for r in db.execute(text("SELECT CASE WHEN run_id LIKE 'run_20261009%' THEN 'new70' ELSE 'old30' END cohort,count(*) evidence,count(distinct attributed_compound_id) compounds FROM auto_validated_evidence GROUP BY 1")).mappings()]
        missing = db.execute(text("SELECT count(*) FROM auto_validated_evidence e WHERE NOT EXISTS (SELECT 1 FROM rag_retrieval_record r WHERE r.evidence_ids ? e.evidence_id::text)")).scalar_one()
        if missing:
            raise RuntimeError("Serving evidence is missing a retrieval index")
        db.rollback()
    engine.dispose()
    result = {"passed": True, "checked_at": ctl.now(), "source_and_keyword_checked": keyword_checked, "raw_files_verified": len(checksums),
              "semantic_compounds": semantic, "serving_counts": counts, "serving_without_rag": missing,
              "embedding_revision": embedding_revision, "ui_started": False}
    ctl.atomic_json(OUT / "retrieval_verification.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
