import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pbpk_kb.acquisition.comprehensive_raw_retriever import (
    ComprehensiveRawRetriever,
    elink_payload_covers_pmids,
    literature_query,
    pmc_official_download_url,
    pmcids_and_pmids_from_elink,
    url_with_params,
)
from pbpk_kb.acquisition.precision import (
    CLINICALTRIAL_CANDIDATE_FIELDS,
    classify_clinical_trial,
    clinicaltrial_candidate_query,
    clinical_trial_has_complete_protocol,
    classify_dailymed_listing,
    classify_openfda_label,
    classify_pubmed_record,
    pubmed_records_from_xml,
)


def test_literature_query_requires_target_and_topic_in_title() -> None:
    query = literature_query("midazolam", ("pharmacokinetic", "AUC"))

    assert "midazolam[Title]" in query
    assert "pharmacokinetic[Title]" in query
    assert "[MeSH Terms]" not in query
    assert "[Supplementary Concept]" not in query


def test_pmc_legacy_ftp_url_uses_current_official_https_layout() -> None:
    original = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/84/11/PMC3603475.tar.gz"

    assert pmc_official_download_url(original) == (
        "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_package/84/11/PMC3603475.tar.gz"
    )


def test_elink_payload_must_cover_every_requested_pmid_without_source_error() -> None:
    complete = {"linksets": [{"ids": ["1"]}, {"ids": ["2"], "linksetdbs": []}]}
    partial = {"linksets": [{"ids": ["1"]}]}
    source_error = {"linksets": [{"ids": ["1"]}, {"ids": ["2"]}], "ERROR": "upstream failure"}

    assert elink_payload_covers_pmids(complete, ["1", "2"]) is True
    assert elink_payload_covers_pmids(partial, ["1", "2"]) is False
    assert elink_payload_covers_pmids(source_error, ["1", "2"]) is False
    assert elink_payload_covers_pmids({}, ["1"]) is False


def test_special_population_query_requires_explicit_pk_support() -> None:
    query = literature_query("midazolam", ("pediatric",), family_id="special_population")

    assert "midazolam[Title]" in query
    assert "pediatric[Title]" in query
    assert "pharmacokinetic[Title/Abstract]" in query


def test_direct_pk_paper_is_selected_for_full_text() -> None:
    result = classify_pubmed_record(
        compound="caffeine",
        title="Pharmacokinetics of caffeine in healthy volunteers",
        abstract="Caffeine clearance and AUC were measured after an oral dose.",
        query_families={"clinical_pk"},
    )

    assert result["decision"] == "fulltext_selected"
    assert result["target_in_title"] is True
    assert "auc" in result["matched_direct_pk_terms"]


def test_food_background_record_is_not_selected_as_adme_full_text() -> None:
    result = classify_pubmed_record(
        compound="caffeine",
        title="Caffeine enemas and toxic cancer breakdown products",
        abstract="Caffeine enemas facilitate excretion of toxic cancer breakdown products.",
        query_families={"adme"},
    )

    assert result["decision"] != "fulltext_selected"
    assert "caffeine enemas" in result["matched_background_terms"]


def test_abstract_only_target_and_generic_exposure_stay_metadata_only() -> None:
    result = classify_pubmed_record(
        compound="midazolam",
        title="Identification of the rate-determining process in atorvastatin clearance",
        abstract="Midazolam was used as a CYP3A probe. Atorvastatin exposure and clearance were measured.",
        query_families={"ddi", "clinical_pk"},
    )

    assert result["decision"] == "metadata_only"
    assert result["selection_reason"] == "target_not_in_title_requires_manual_role_review"


def test_target_title_without_strong_pk_topic_stays_metadata_only() -> None:
    result = classify_pubmed_record(
        compound="midazolam",
        title="Low-dose midazolam attenuates odor avoidance in rats",
        abstract="The behavioral effect was associated with systemic exposure.",
        query_families={"ddi"},
    )

    assert result["decision"] == "metadata_only"
    assert "exposure" not in result["matched_strong_pbpk_terms"]


def test_special_population_title_with_abstract_pk_support_is_selected() -> None:
    result = classify_pubmed_record(
        compound="midazolam",
        title="Midazolam pharmacokinetics in pediatric patients",
        abstract="AUC and clearance were compared across age groups.",
        query_families={"special_population"},
    )

    assert result["decision"] == "fulltext_selected"
    assert result["selection_reason"] == "exact_target_in_title_and_strong_pk_topic"


def test_combination_dailymed_listing_is_metadata_only_by_default() -> None:
    result = classify_dailymed_listing(
        compound="caffeine",
        row={"setid": "set-1", "title": "Butalbital, Aspirin, Caffeine, and Codeine Capsules"},
        include_combination_labels=False,
    )

    assert result["decision"] == "metadata_only"
    assert result["is_combination"] is True
    assert result["identity_match_type"] == "target_component"


def test_combination_openfda_label_is_metadata_only() -> None:
    result = classify_openfda_label(
        "caffeine",
        {
            "id": "label-1",
            "openfda": {"generic_name": ["Butalbital, Aspirin, Caffeine and Codeine"]},
        },
    )

    assert result["decision"] == "metadata_only"
    assert result["identity_match_type"] == "target_component"


def test_pk_intervention_trial_is_selected_but_background_mention_is_not() -> None:
    selected = classify_clinical_trial(
        "midazolam",
        {
            "protocolSection": {
                "identificationModule": {"briefTitle": "Midazolam pharmacokinetic study"},
                "descriptionModule": {"briefSummary": "AUC and clearance will be measured."},
                "armsInterventionsModule": {"interventions": [{"name": "Midazolam"}]},
            }
        },
    )
    excluded = classify_clinical_trial(
        "midazolam",
        {
            "protocolSection": {
                "identificationModule": {"briefTitle": "Sleep quality study"},
                "descriptionModule": {"briefSummary": "Participants may report prior midazolam use."},
                "armsInterventionsModule": {"interventions": [{"name": "Placebo"}]},
            }
        },
    )

    assert selected["decision"] == "fulltext_selected"
    assert excluded["decision"] == "excluded"


def test_field_limited_clinicaltrials_candidate_is_not_a_complete_study_record() -> None:
    complete = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "statusModule": {"overallStatus": "COMPLETED"},
            "designModule": {"studyType": "INTERVENTIONAL"},
        }
    }
    field_limited_candidate = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000002", "briefTitle": "Midazolam PK study"},
            "descriptionModule": {"briefSummary": "AUC is measured."},
            "armsInterventionsModule": {"interventions": [{"name": "Midazolam"}]},
        }
    }

    assert clinical_trial_has_complete_protocol(complete) is True
    assert clinical_trial_has_complete_protocol(field_limited_candidate) is False


def test_clinicaltrials_candidate_query_and_fields_are_precision_limited() -> None:
    query = clinicaltrial_candidate_query()

    assert "pharmacokinetic" in query
    assert "drug interaction" in query
    assert CLINICALTRIAL_CANDIDATE_FIELDS == "NCTId|BriefTitle|OfficialTitle|BriefSummary|InterventionName"


def test_clinicaltrials_downloads_full_json_only_for_selected_field_limited_candidates() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        candidate_path = root / "clinicaltrials_candidate.json"
        candidate_path.write_text(
            json.dumps(
                {
                    "studies": [
                        {
                            "protocolSection": {
                                "identificationModule": {
                                    "nctId": "NCT00000001",
                                    "briefTitle": "Midazolam pharmacokinetic study",
                                },
                                "descriptionModule": {"briefSummary": "Midazolam AUC and clearance are measured."},
                                "armsInterventionsModule": {"interventions": [{"name": "Midazolam"}]},
                            }
                        },
                        {
                            "protocolSection": {
                                "identificationModule": {
                                    "nctId": "NCT00000002",
                                    "briefTitle": "Midazolam sleep study",
                                },
                                "descriptionModule": {"briefSummary": "Sleep quality is measured."},
                                "armsInterventionsModule": {"interventions": [{"name": "Midazolam"}]},
                            }
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        retriever = ComprehensiveRawRetriever(
            output_root=root,
            compounds=["midazolam"],
            source_registry_path=Path("pbpk_kb/config/sources.yaml"),
            timeout_seconds=1,
            delay_seconds=0,
            ncbi_batch_size=1,
            page_size=1,
            dailymed_workers=1,
            pkdb_workers=1,
            download_oa_packages=False,
            mirror_repositories=False,
            include_shared_sources=False,
            include_combination_labels=False,
            max_pmc_fulltext_per_compound=1,
            max_pmc_oa_packages_per_compound=1,
            official_run=False,
            clinicaltrials_workers=1,
        )
        fetch_calls: list[dict[str, object]] = []
        detail_calls: list[tuple[str, str, str]] = []

        def fake_fetch_and_store(**kwargs: object) -> dict[str, str]:
            fetch_calls.append(kwargs)
            return {
                "storage_path": str(candidate_path),
                "url": "https://clinicaltrials.gov/api/v2/studies?candidate=true",
                "request_id": "clinicaltrials:candidate:1",
            }

        def fake_detail(compound: str, nct_id: str, *, source_request_id: str) -> None:
            detail_calls.append((compound, nct_id, source_request_id))

        retriever.fetch_and_store = fake_fetch_and_store  # type: ignore[method-assign]
        retriever.retrieve_clinicaltrial_detail = fake_detail  # type: ignore[method-assign]
        retriever.retrieve_clinicaltrials_pages("midazolam")

        assert fetch_calls[0]["group"] == "clinicaltrials/candidate_pages"
        assert fetch_calls[0]["params"] == {
            "query.intr": "midazolam",
            "query.term": clinicaltrial_candidate_query(),
            "fields": CLINICALTRIAL_CANDIDATE_FIELDS,
            "pageSize": 100,
            "format": "json",
        }
        assert detail_calls == [("midazolam", "NCT00000001", "clinicaltrials:candidate:1")]


def test_pubmed_xml_parser_preserves_title_and_abstract() -> None:
    payload = b"""
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation><PMID>123</PMID><Article>
          <ArticleTitle>Midazolam PK</ArticleTitle>
          <Abstract><AbstractText Label='RESULTS'>AUC increased.</AbstractText></Abstract>
        </Article></MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    assert pubmed_records_from_xml(payload) == [
        {"pmid": "123", "title": "Midazolam PK", "abstract": "RESULTS: AUC increased.", "publication_types": []}
    ]


def test_url_encoding_repeats_elink_ids() -> None:
    url = url_with_params("https://example.test/elink", {"id": ["1", "2"], "db": "pmc"})

    assert "id=1&id=2" in url


def test_elink_mapping_and_pmc_rank_preserve_pubmed_selection_priority() -> None:
    pmcids, pmids, mapping = pmcids_and_pmids_from_elink(
        {
            "linksets": [
                {"ids": ["100"], "linksetdbs": [{"links": ["10"]}]},
                {"ids": ["200"], "linksetdbs": [{"links": ["20"]}]},
            ]
        }
    )
    assert pmcids == {"PMC10", "PMC20"}
    assert pmids == {"100", "200"}
    assert mapping["PMC10"] == {"100"}

    with TemporaryDirectory() as directory:
        retriever = ComprehensiveRawRetriever(
            output_root=Path(directory),
            compounds=["midazolam"],
            source_registry_path=Path("pbpk_kb/config/sources.yaml"),
            timeout_seconds=1,
            delay_seconds=0,
            ncbi_batch_size=1,
            page_size=1,
            dailymed_workers=1,
            pkdb_workers=1,
            download_oa_packages=False,
            mirror_repositories=False,
            include_shared_sources=False,
            include_combination_labels=False,
            max_pmc_fulltext_per_compound=1,
            max_pmc_oa_packages_per_compound=1,
            official_run=False,
        )
        retriever.pubmed_selection_scores_by_pmid["midazolam"].update({"100": 6, "200": 9})
        retriever.pmcid_to_pmids_by_compound["midazolam"].update(mapping)
        assert retriever.ranked_pmcids("midazolam", pmcids) == ["PMC20", "PMC10"]


def test_compound_runs_and_shared_snapshots_are_separate() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        retriever = ComprehensiveRawRetriever(
            output_root=root,
            compounds=["midazolam"],
            source_registry_path=Path("pbpk_kb/config/sources.yaml"),
            timeout_seconds=1,
            delay_seconds=0,
            ncbi_batch_size=1,
            page_size=1,
            dailymed_workers=1,
            pkdb_workers=1,
            download_oa_packages=False,
            mirror_repositories=False,
            include_shared_sources=False,
            include_combination_labels=False,
            max_pmc_fulltext_per_compound=1,
            max_pmc_oa_packages_per_compound=1,
            official_run=False,
        )

        assert retriever.run_root == root / "compound_runs" / retriever.run_id
        assert retriever.shared_snapshot_root("osp_pbpk_model_library") == root / "source_cache" / "osp_pbpk_model_library" / retriever.run_id


def test_selection_manifest_has_provenance_fields() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        retriever = ComprehensiveRawRetriever(
            output_root=root,
            compounds=["midazolam"],
            source_registry_path=Path("pbpk_kb/config/sources.yaml"),
            timeout_seconds=1,
            delay_seconds=0,
            ncbi_batch_size=1,
            page_size=1,
            dailymed_workers=1,
            pkdb_workers=1,
            download_oa_packages=False,
            mirror_repositories=False,
            include_shared_sources=False,
            include_combination_labels=False,
            max_pmc_fulltext_per_compound=1,
            max_pmc_oa_packages_per_compound=1,
            official_run=False,
        )
        retriever.record_selection(
            source_id="pubmed",
            compound="midazolam",
            source_record_id="123",
            decision={
                "decision": "fulltext_selected",
                "selection_reason": "exact_target_and_direct_pk_topic",
                "selection_score": 8,
                "query_families": ["clinical_pk"],
                "identity_match_type": "exact_target",
            },
            record={"pmid": "123"},
            source_url="https://example.test/efetch?id=123",
            source_location="pmid:123",
            request_id="pubmed:123",
        )

        path = retriever.run_root / "compounds" / "midazolam" / "selection" / "pubmed" / "candidate_records.jsonl"
        row = json.loads(path.read_text(encoding="utf-8").strip())
        assert row["query_family"] == ["clinical_pk"]
        assert row["identity_match_type"] == "exact_target"
        assert row["source_url"] == "https://example.test/efetch?id=123"
        assert row["request_id"] == "pubmed:123"


def test_identity_and_shared_snapshot_manifests_are_raw_only() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        retriever = ComprehensiveRawRetriever(
            output_root=root,
            compounds=["midazolam"],
            source_registry_path=Path("pbpk_kb/config/sources.yaml"),
            timeout_seconds=1,
            delay_seconds=0,
            ncbi_batch_size=1,
            page_size=1,
            dailymed_workers=1,
            pkdb_workers=1,
            download_oa_packages=False,
            mirror_repositories=False,
            include_shared_sources=False,
            include_combination_labels=False,
            max_pmc_fulltext_per_compound=1,
            max_pmc_oa_packages_per_compound=1,
            official_run=False,
        )
        retriever.write_identity_coverage_manifest("midazolam", [])
        snapshot_root = retriever.shared_snapshot_root("osp_pbpk_model_library")
        retriever.write_shared_snapshot_manifest(
            {"source_id": "osp_pbpk_model_library", "source_name": "OSP", "access_url": "https://example.test/repo"},
            "repository_mirror",
            snapshot_root,
        )

        identity = json.loads(
            (retriever.run_root / "compounds" / "midazolam" / "identity" / "identity_coverage_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        snapshot = json.loads((snapshot_root / "snapshot_manifest.json").read_text(encoding="utf-8"))
        assert identity["status"] == "raw_identity_discovery_only"
        assert identity["raw_only"] is True
        assert snapshot["source_version"] == f"snapshot:{retriever.run_id}"
        assert snapshot["raw_only"] is True


def test_pubmed_open_access_gap_manifest_separates_missing_links_from_lookup_failures() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        retriever = ComprehensiveRawRetriever(
            output_root=root,
            compounds=["midazolam"],
            source_registry_path=Path("pbpk_kb/config/sources.yaml"),
            timeout_seconds=1,
            delay_seconds=0,
            ncbi_batch_size=20,
            page_size=1,
            dailymed_workers=1,
            pkdb_workers=1,
            download_oa_packages=False,
            mirror_repositories=False,
            include_shared_sources=False,
            include_combination_labels=False,
            max_pmc_fulltext_per_compound=1,
            max_pmc_oa_packages_per_compound=1,
            official_run=False,
        )
        retriever.selected_pmids_by_compound["midazolam"].update({"1", "2", "3"})
        retriever.pubmed_pmc_link_lookup_succeeded_pmids_by_compound["midazolam"].update({"1", "2"})
        retriever.pubmed_pmids_with_pmcid_by_compound["midazolam"].add("1")
        retriever.write_pubmed_open_access_gap_manifest("midazolam")

        path = retriever.run_root / "compounds" / "midazolam" / "pubmed" / "open_access_gap_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["selected_pmids_confirmed_without_pmc_open_fulltext_link_list"] == ["2"]
        assert payload["selected_pmids_with_pmc_link_lookup_failure_list"] == ["3"]


def test_raw_request_retry_preserves_attempt_specific_request_keys() -> None:
    with TemporaryDirectory() as directory:
        retriever = ComprehensiveRawRetriever(
            output_root=Path(directory),
            compounds=["midazolam"],
            source_registry_path=Path("pbpk_kb/config/sources.yaml"),
            timeout_seconds=1,
            delay_seconds=0,
            ncbi_batch_size=20,
            page_size=1,
            dailymed_workers=1,
            pkdb_workers=1,
            download_oa_packages=False,
            mirror_repositories=False,
            include_shared_sources=False,
            include_combination_labels=False,
            max_pmc_fulltext_per_compound=1,
            max_pmc_oa_packages_per_compound=1,
            official_run=False,
        )
        request_keys: list[str] = []

        def fake_fetch_and_store(**kwargs: object) -> dict[str, str] | None:
            request_keys.append(str(kwargs["request_key"]))
            return {"request_id": "success"} if len(request_keys) == 3 else None

        retriever.fetch_and_store = fake_fetch_and_store  # type: ignore[method-assign]
        asset = retriever.fetch_and_store_with_retries(
            source_id="pubmed",
            compound="midazolam",
            group="pubmed/test",
            request_key="efetch_test",
            url="https://example.test/efetch",
            retry_delay_seconds=0,
        )

        assert asset == {"request_id": "success"}
        assert request_keys == ["efetch_test", "efetch_test_retry_1", "efetch_test_retry_2"]


def test_deferred_limit_record_does_not_inflate_failure_count() -> None:
    with TemporaryDirectory() as directory:
        retriever = ComprehensiveRawRetriever(
            output_root=Path(directory),
            compounds=["midazolam"],
            source_registry_path=Path("pbpk_kb/config/sources.yaml"),
            timeout_seconds=1,
            delay_seconds=0,
            ncbi_batch_size=20,
            page_size=1,
            dailymed_workers=1,
            pkdb_workers=1,
            download_oa_packages=False,
            mirror_repositories=False,
            include_shared_sources=False,
            include_combination_labels=False,
            max_pmc_fulltext_per_compound=1,
            max_pmc_oa_packages_per_compound=1,
            official_run=False,
        )
        retriever.log_deferred(
            "pmc_oa:threshold",
            "pmc_oa",
            "midazolam",
            "pmc/fulltext_threshold",
            "pmc_fulltext_selection",
            "Configured test limit deferred records.",
        )
        summary = retriever.write_summary()

        assert summary["failure_count"] == 0
        assert summary["deferred_count"] == 1
        assert json.loads(retriever.deferred_path.read_text(encoding="utf-8"))["status"] == "deferred"
