import json
from hashlib import sha256
from pathlib import Path

import pytest

from pbpk_kb.acquisition.reconcile_final_acquisition import reconcile


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_reconcile_marks_fully_covered_run_complete(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    run_root = output_root / "compound_runs" / "run_test"
    raw_path = run_root / "shared" / "pmc_static_objects" / "PMC123" / "pmc_oa" / "article.xml"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("<article />", encoding="utf-8")
    metadata_path = raw_path.with_suffix(".xml.metadata.json")
    metadata_path.write_text("{}", encoding="utf-8")

    plan = {
        "source_id": "pmc_oa",
        "pmcid": "PMC123",
        "s3_key": "PMC123.1/article.xml",
        "url": "https://pmc-oa-opendata.s3.amazonaws.com/PMC123.1/article.xml",
        "byte_count": raw_path.stat().st_size,
        "status": "deferred",
        "raw_only": True,
    }
    asset = {
        "asset_id": "asset:1",
        "request_id": "pmc_oa:static_object_1",
        "source_id": "pmc_oa",
        "compound": None,
        "group": "pmc/oa_s3_objects/PMC123",
        "request_key": "static_object_1",
        "url": plan["url"],
        "storage_path": str(raw_path),
        "metadata_path": str(metadata_path),
        "checksum": sha256(raw_path.read_bytes()).hexdigest(),
        "byte_count": raw_path.stat().st_size,
        "metadata": {"pmcid": "PMC123", "s3_key": plan["s3_key"]},
        "raw_only": True,
    }
    write_jsonl(run_root / "assets.jsonl", [asset])
    write_jsonl(run_root / "failures.jsonl", [])
    write_jsonl(run_root / "finalization" / "pmc_object_download_plan.jsonl", [plan])
    write_json(run_root / "download_plan.json", {})
    write_json(
        run_root / "compounds" / "midazolam" / "pmc_oa" / "pmc_fulltext_gap_manifest.json",
        {"selected_for_fulltext_download": ["PMC123"], "skipped_pmcids": []},
    )
    write_json(
        run_root / "summary.json",
        {
            "run_id": "run_test",
            "compound_count": 1,
            "asset_count": 0,
            "failure_count": 0,
            "pmid_count_by_compound": {"midazolam": 1},
            "coverage_by_source": {},
            "openfda_selection_by_compound": {"midazolam": {"fulltext_selected": 2}},
        },
    )
    write_json(run_root / "RUN_STATUS.json", {"status": "completed_with_deferred_pmc_object_transfer"})

    result = reconcile(run_root, output_root, ["midazolam"])

    assert result["status"] == "completed"
    assert result["accepted_as_complete_raw_content"] is True
    assert result["pmc_object_transfer"]["completed_object_count"] == 1
    assert result["pmc_article_coverage"]["incomplete_article_count"] == 0
    assert result["source_payload_validation"]["unrecovered_payload_count"] == 0
    assert result["integrity_audit"]["status"] == "passed"
    assert result["integrity_audit"]["issue_count"] == 0
    assert (run_root / "finalization" / "FINAL_INTEGRITY_AUDIT.json").exists()
    assert (run_root / "finalization" / "pubmed_elink_payload_validation.jsonl").exists()
    assert json.loads((run_root / "RUN_STATUS.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert json.loads((run_root / "summary.json").read_text(encoding="utf-8"))["asset_count"] == 1
    assert json.loads((run_root / "summary.json").read_text(encoding="utf-8"))["openfda_selection_by_compound"] == {
        "midazolam": {"fulltext_selected": 2}}
    ledger = run_root / "finalization" / "failure_resolutions.jsonl"
    assert ledger.read_bytes() == b""
    assert result["failure_resolution_count"] == 0
    # Real downstream inventory must accept this zero-failure output without
    # weakening its requirement that the explicit ledger actually exists.
    from pbpk_kb.processing.raw_processor import RawProcessingPipeline
    pipeline = RawProcessingPipeline(run_root=run_root, output_root=tmp_path / "processed",
                                     stage="inventory", llm_mode="off", embedding_mode="off")
    inventory = pipeline.process()
    assert inventory["accepted"] is True
    assert inventory["inventory_validation_counts"]["checksum_verified"] == 1
    ledger.unlink()
    with pytest.raises(FileNotFoundError, match="failure_resolutions"):
        pipeline._validate_run()


def test_new_drug_batch_does_not_inherit_previous_drug_model_snapshot(tmp_path):
    from pbpk_kb.acquisition.reconcile_final_acquisition import completed_run_osp_snapshot
    old = tmp_path / "source_cache/osp_pbpk_model_library/run_old/verified"
    write_json(old / "RUN_STATUS.json", {"status": "completed"})
    write_json(old / "snapshot_manifest.json", {"compounds": ["caffeine"]})
    assert completed_run_osp_snapshot(tmp_path, "run_new") is None
    assert completed_run_osp_snapshot(tmp_path, "run_old") == old
    new = tmp_path / "source_cache/osp_pbpk_model_library/run_new"
    write_json(new / "RUN_STATUS.json", {"status": "running"})
    write_json(new / "snapshot_manifest.json", {"compounds": ["atorvastatin"]})
    assert completed_run_osp_snapshot(tmp_path, "run_new") is None
    write_json(new / "RUN_STATUS.json", {"status": "completed"})
    assert completed_run_osp_snapshot(tmp_path, "run_new") == new
