import json
from hashlib import sha256
from pathlib import Path

import pytest

from pbpk_kb.acquisition.comprehensive_raw_retriever import ComprehensiveRawRetriever
from pbpk_kb.acquisition.finalize_raw_run import s3_common_prefixes, s3_contents
from pbpk_kb.acquisition.reconcile_final_acquisition import confirmed_absent_from_public_cloud


def test_invalid_or_truncated_s3_responses_are_not_empty_dataset(tmp_path):
    path = tmp_path / "listing.xml"
    path.write_text("<Error><Code>AccessDenied</Code></Error>")
    with pytest.raises(ValueError):
        s3_common_prefixes(path)
    with pytest.raises(ValueError):
        s3_contents(path)
    path.write_text("<ListBucketResult><IsTruncated>true</IsTruncated></ListBucketResult>")
    with pytest.raises(ValueError):
        s3_contents(path)


def test_absence_requires_exact_prefix_checksum_and_complete_empty_result(tmp_path):
    path = tmp_path / "listing.xml"
    path.write_text("<ListBucketResult><Prefix>PMC123.</Prefix><KeyCount>0</KeyCount><IsTruncated>false</IsTruncated></ListBucketResult>")
    asset = {"request_key": "oa_s3_versions_PMC123", "storage_path": str(path),
             "url": "https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&prefix=PMC123.&delimiter=/",
             "checksum": sha256(path.read_bytes()).hexdigest()}
    assert confirmed_absent_from_public_cloud(asset, "PMC123")
    assert not confirmed_absent_from_public_cloud(asset, "PMC124")
    path.write_text("<ListBucketResult/>")
    assert not confirmed_absent_from_public_cloud(asset, "PMC123")


def test_current_discovery_never_calls_retired_oa_api():
    retriever = object.__new__(ComprehensiveRawRetriever)
    calls = []
    retriever.fetch_and_store = lambda **kwargs: calls.append(kwargs)
    retriever.retrieve_pmc_oa_package("caffeine", "PMC123")
    assert len(calls) == 1
    assert calls[0]["url"] == "https://pmc-oa-opendata.s3.amazonaws.com/"
    assert calls[0]["params"]["prefix"] == "PMC123."


def test_resume_restores_links_without_redownloading_or_changing_raw(tmp_path):
    source = tmp_path / "sources.yaml"
    source.write_text("- source_id: pubmed\n  enabled: true\n")
    retriever = ComprehensiveRawRetriever(output_root=tmp_path, compounds=["caffeine"],
        source_registry_path=source, timeout_seconds=10, delay_seconds=0, ncbi_batch_size=20,
        page_size=100, dailymed_workers=1, pkdb_workers=1, download_oa_packages=True,
        mirror_repositories=False, include_shared_sources=False, include_combination_labels=False,
        max_pmc_fulltext_per_compound=250, max_pmc_oa_packages_per_compound=100, official_run=True)
    run = tmp_path / "compound_runs/run_test"
    with pytest.raises(RuntimeError, match="Discovery is unfinished"):
        retriever.resume_after_discovery(run)
    base = run / "compounds/caffeine"
    (base / "pubmed").mkdir(parents=True)
    (base / "pubmed/open_access_gap_manifest.json").write_text("{}")
    link = base / "link.json"
    link.write_text(json.dumps({"linksets": [{"ids": ["10"], "linksetdbs": [{"linkname": "pubmed_pmc", "links": ["123"]}]}]}))
    asset = {"asset_id": "a1", "request_id": "r1", "compound": "caffeine", "source_id": "pubmed",
             "storage_path": str(link), "checksum": sha256(link.read_bytes()).hexdigest(),
             "group": "pubmed/elink_pmc", "byte_count": link.stat().st_size}
    (run / "assets.jsonl").write_text(json.dumps(asset) + "\n")
    before = link.read_bytes()
    retriever.retrieve_pmc_fulltext = lambda: None
    retriever.write_summary = lambda: {}
    retriever.write_status = lambda *args, **kwargs: None
    retriever.resume_after_discovery(run)
    assert retriever.pmcids_by_compound["caffeine"] == {"PMC123"}
    assert link.read_bytes() == before
    assert retriever._resume_assets["r1"] == asset
    retriever.request_id = lambda *args: "r1"
    retriever.fetch_bytes = lambda url: pytest.fail("Verified raw asset must not be downloaded again")
    assert retriever.fetch_and_store(source_id="pubmed", compound="caffeine", group="x", request_key="x", url="https://example.invalid") == asset


def test_explicit_new_run_binds_every_ledger_and_refuses_populated_or_external_path(tmp_path):
    retriever = object.__new__(ComprehensiveRawRetriever)
    retriever.output_root = tmp_path / "raw"
    fresh = retriever.output_root / "compound_runs/run_named"
    fresh.mkdir(parents=True)
    retriever.bind_new_run_root(fresh)
    assert retriever.run_id == "run_named"
    assert retriever.requests_path == fresh / "requests.jsonl"
    assert retriever.status_path == fresh / "RUN_STATUS.json"
    (fresh / "asset.bin").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="must be empty"):
        retriever.bind_new_run_root(fresh)
    with pytest.raises(ValueError, match="inside"):
        retriever.bind_new_run_root(tmp_path / "external_run")
    assert (fresh / "asset.bin").read_bytes() == b"preserve"
