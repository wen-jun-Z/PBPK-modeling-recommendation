"""Structured evidence must resolve real raw sources without masking broken links."""
import pytest
from pbpk_kb.api.evidence_routes import source_document_for_processed_row


class SourceSession:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def execute(self, statement, params):
        self.calls.append((str(statement), params))
        return self

    def mappings(self):
        return self

    def first(self):
        return self.result


@pytest.mark.parametrize("source,locator", [
    ("pubchem", "record#json_path=$.Properties[0].ConnectivitySMILES"),
    ("dailymed", "label#table=2;row=3;column=4"),
])
def test_structured_and_recovered_table_sources_resolve_imported_raw(source, locator):
    raw = {"raw_document_id": "raw-id", "source_record_id": "asset:1", "source_id": source,
           "raw_storage_path": "/project/raw", "raw_checksum": "abc", "source_url": "https://example.test/source"}
    session = SourceSession(raw)
    result = source_document_for_processed_row(session, {"raw_document_id": "raw-id",
        "raw_asset_id": "asset:1", "source_id": source, "source_location": locator})
    assert result["source_location"] == locator
    assert result["raw_storage_path"] == raw["raw_storage_path"]
    assert result["raw_checksum"] == raw["raw_checksum"]
    assert result["source_kind"] == "raw_asset" and result["parsed_document_id"] is None
    sql, params = session.calls[0]
    assert "source_record_id = :raw_asset_id" in sql and "source_id = :source_id" in sql
    assert params == {"raw_document_id": "raw-id", "raw_asset_id": "asset:1", "source_id": source}


def test_nonexistent_raw_source_does_not_use_unverified_inline_fields():
    session = SourceSession(None)
    assert source_document_for_processed_row(session, {"raw_document_id": "absent", "raw_asset_id": "asset:1",
        "source_id": "pubchem", "source_location": "somewhere", "storage_path": "/fake", "checksum": "fake"}) is None


def test_article_source_keeps_existing_path():
    session = SourceSession({"parsed_document_id": "article", "title": "Existing article"})
    assert source_document_for_processed_row(session, {"parsed_document_id": "article"}) == session.result
    assert session.calls[0][1] == {"parsed_document_id": "article"}


def test_broken_article_link_is_not_hidden_by_raw_fallback():
    session = SourceSession(None)
    assert source_document_for_processed_row(session, {"parsed_document_id": "absent", "raw_document_id": "raw-id",
        "raw_asset_id": "asset:1", "source_id": "pubmed"}) is None
    assert len(session.calls) == 1


def test_absent_source_identity_still_returns_none():
    session = SourceSession({"raw_document_id": "unrelated"})
    assert source_document_for_processed_row(session, {"raw_document_id": "raw-id"}) is None
    assert not session.calls
