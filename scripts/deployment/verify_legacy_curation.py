"""Exercise real keyword/detail/semantic retrieval after serving curation."""
from collections import defaultdict
import json

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

import curate_legacy_evidence as curate
from verify_expansion_batch import (_registered_database, _local_query_vectors, _semantic_with_local_vectors,
                                    _verify_source_detail, _verify_semantic_hit)
from pbpk_kb.api.search_routes import search_evidence, semantic_search
from pbpk_kb.api.evidence_routes import evidence_detail


def main():
    bundle = curate.read(curate.BASE / "apply_bundle.json")
    url, config = _registered_database()
    engine = create_engine(url, connect_args={"options": "-c default_transaction_read_only=on -c statement_timeout=30000"})
    expected = set(bundle["scope"]["active_ids"])
    checksums, per_compound = {}, []
    with Session(engine) as db:
        rows = list(db.execute(text("SELECT to_jsonb(t) FROM auto_validated_evidence t WHERE evidence_id::text=ANY(:ids)"), {"ids": sorted(expected)}).scalars())
        assert {r["evidence_id"] for r in rows} == expected
        grouped = defaultdict(list)
        for row in rows:
            detail = evidence_detail(row["evidence_type"], row["evidence_id"], db=db)
            _verify_source_detail(detail, checksums)
            grouped[row["attributed_compound_name"]].append(row)
        rag = list(db.execute(text("SELECT retrieval_record_id::text,compound_name,evidence_ids,embedding_text,source_id FROM rag_retrieval_record WHERE retrieval_record_id::text=ANY(:ids)"), {"ids": bundle["scope"]["new_rag_ids"]}).mappings())
        queries = []
        for drug, candidates in sorted(grouped.items()):
            sample = next((r for r in candidates if r["value"] is not None), candidates[0])
            hits = search_evidence(compound_id=sample["attributed_compound_id"], q=sample["parameter"], limit=500, offset=0, min_confidence=0, db=db)
            assert sample["evidence_id"] in {h["evidence_id"] for h in hits}, drug
            selected = next(r for r in rag if sample["evidence_id"] in r["evidence_ids"])
            queries.append(selected["embedding_text"])
            per_compound.append({"compound": drug, "keyword_passed": True, "retrieval_record_id": selected["retrieval_record_id"], "source_id": selected["source_id"]})
        vectors, revision = _local_query_vectors(queries, config)
        local_semantic = _semantic_with_local_vectors(semantic_search, vectors)
        for result, query in zip(per_compound, queries):
            hits = local_semantic(q=query, source_id=result["source_id"], limit=100, db=db)
            result["semantic"] = _verify_semantic_hit(hits, {result["retrieval_record_id"]})
        animal = db.execute(text("SELECT evidence_id::text,attributed_compound_id::text,source_record->'species' AS species FROM auto_validated_evidence WHERE jsonb_typeof(source_record->'species')='array' AND jsonb_array_length(source_record->'species')>0 LIMIT 1")).mappings().first()
        if animal:
            hits = search_evidence(compound_id=animal["attributed_compound_id"], species=animal["species"][0], limit=500, offset=0, min_confidence=0, db=db)
            assert animal["evidence_id"] in {h["evidence_id"] for h in hits}
        missing = db.execute(text("SELECT count(*) FROM auto_validated_evidence e WHERE NOT EXISTS (SELECT 1 FROM rag_retrieval_record r WHERE r.evidence_ids ? e.evidence_id::text)")).scalar_one()
        counts = list(db.execute(text("SELECT CASE WHEN run_id LIKE 'run_20261009%' THEN 'new70' ELSE 'old30' END cohort,count(*) evidence,count(distinct attributed_compound_id) compounds FROM auto_validated_evidence GROUP BY 1")).mappings())
        db.rollback()
    engine.dispose()
    result = {"passed": True, "raw_files_verified": len(checksums), "all_active_curation_details_checked": len(rows),
              "keyword_and_semantic_compounds": len(per_compound), "compound_checks": per_compound,
              "serving_counts": [dict(x) for x in counts], "serving_evidence_without_rag": missing,
              "species_filter_verified": bool(animal), "embedding_revision": revision,
              "external_service_used": False, "ui_started": False}
    curate.atomic(curate.BASE / "retrieval_verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "compound_checks"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
