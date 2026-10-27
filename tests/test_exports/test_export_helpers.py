from pbpk_kb.exports.common import pg_enum_array_literal, safe_slug
from pbpk_kb.exports.compound_profile_export import build_compound_profile_markdown
from pbpk_kb.exports.stage1_report_export import build_stage1_summary_markdown


def test_pg_enum_array_literal_quotes_status_values() -> None:
    assert pg_enum_array_literal(["needs_review", "approved_for_modeling"]) == '{"needs_review","approved_for_modeling"}'
    assert pg_enum_array_literal([]) == "{}"


def test_safe_slug_has_stable_fallback() -> None:
    assert safe_slug("Midazolam HCl") == "midazolam-hcl"
    assert safe_slug("!!!", fallback="all") == "all"


def test_record_export_job_source_contains_activity_log_hook() -> None:
    common_source = __import__("inspect").getsource(__import__("pbpk_kb.exports.common", fromlist=["record_export_job"]).record_export_job)

    assert "record_user_activity" in common_source
    assert "export_created" in common_source


def test_compound_profile_markdown_includes_provenance_and_status() -> None:
    markdown = build_compound_profile_markdown(
        compound={"preferred_name": "Midazolam", "source_id": "pubchem", "source_location": "CID:4192", "curation_status": "machine_extracted"},
        evidence_rows=[
            {
                "evidence_type": "clinical_pk",
                "parameter_name": "clearance",
                "value_text": "9.3 to 11.0",
                "unit": "mL/min/kg",
                "source_id": "dailymed",
                "source_location": "SPL_SET_ID:x",
                "curation_status": "needs_review",
            }
        ],
        model_rows=[],
        sources={"dailymed": {"source_version": "2026-06-13", "checksum": "abc"}},
        filters={"compound": "midazolam"},
    )

    assert "# Compound Profile Export: Midazolam" in markdown
    assert "dailymed" in markdown
    assert "SPL_SET_ID:x" in markdown
    assert "needs_review" in markdown


def test_stage1_summary_reports_rag_records_and_missing_snapshots_explicitly() -> None:
    markdown = build_stage1_summary_markdown(
        metrics={"document_chunk_count": 0, "rag_record_count": 3342},
        source_counts=[],
        sources={
            "pubmed": {"source_version": "run_20260713T163840Z", "checksum": "abc"},
            "rxnorm": {"source_version": None, "checksum": None},
        },
    )

    assert "rag_record_count: 3342" in markdown
    assert "rxnorm: version=no current snapshot" in markdown
    assert "checksum=no current snapshot checksum" in markdown

