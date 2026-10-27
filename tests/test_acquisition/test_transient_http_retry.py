import json
from contextlib import nullcontext
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from pbpk_kb.acquisition import comprehensive_raw_retriever as retrieval


URL = "https://api.fda.gov/drug/label.json?search=sildenafil"


@pytest.fixture
def retriever(tmp_path, monkeypatch):
    registry = tmp_path / "sources.yaml"
    registry.write_text("")
    instance = retrieval.ComprehensiveRawRetriever(
        output_root=tmp_path, compounds=["sildenafil"], source_registry_path=registry,
        timeout_seconds=1, delay_seconds=0, ncbi_batch_size=1, page_size=1,
        dailymed_workers=1, pkdb_workers=1, download_oa_packages=False,
        mirror_repositories=False, include_shared_sources=False,
        include_combination_labels=False, max_pmc_fulltext_per_compound=1,
        max_pmc_oa_packages_per_compound=1, official_run=False,
    )
    monkeypatch.setattr(instance, "fetch_bytes_with_curl", lambda *a, **k: pytest.fail("Unexpected curl fallback"))
    return instance


def fake_responses(monkeypatch, outcomes):
    calls, delays = [], []
    outcomes = iter(outcomes)

    def open_response(request, **kwargs):
        calls.append(request.full_url)
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return nullcontext(SimpleNamespace(read=lambda: outcome, headers={"content-type": "application/json"}, status=200))

    monkeypatch.setattr(retrieval, "urlopen", open_response)
    monkeypatch.setattr(retrieval.time, "sleep", delays.append)
    return calls, delays


def error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(URL, code, "provider error", headers, None)


def events(instance):
    path = instance.run_root / "transport_attempts.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_transient_error_then_success_is_audited_without_failed_request(retriever, monkeypatch, code):
    calls, delays = fake_responses(monkeypatch, [error(code), b'{"results": []}'])
    asset = retriever.fetch_and_store(
        source_id="openfda_drug_label", compound="sildenafil", group="openfda_drug_label",
        request_key="label_skip_0", url=URL,
    )
    assert len(calls) == 2
    assert delays == [1.0]
    assert asset["status_code"] == 200
    assert Path(asset["storage_path"]).read_bytes() == b'{"results": []}'
    assert not retriever.failures_path.exists()
    assert [row["status"] for row in map(json.loads, retriever.requests_path.read_text().splitlines())] == ["started", "fetched"]
    rows = events(retriever)
    assert len(rows) == 1
    assert rows[0]["url"] == URL and rows[0]["status_code"] == code
    assert rows[0]["attempt"] == 1 and rows[0]["max_attempts"] == 3
    assert rows[0]["timestamp"] and rows[0]["status"] == "retry_pending"


def test_three_transient_failures_raise_without_fourth_attempt(retriever, monkeypatch):
    calls, delays = fake_responses(monkeypatch, [error(500), error(500), error(500)])
    with pytest.raises(HTTPError) as caught:
        retriever.fetch_bytes(URL)
    assert caught.value.code == 500
    assert len(calls) == 3 and delays == [1.0, 2.0]
    assert [row["attempt"] for row in events(retriever)] == [1, 2, 3]
    assert events(retriever)[-1]["status"] == "retry_exhausted"
    assert events(retriever)[-1]["retry_delay_seconds"] == 0


def test_exhausted_request_still_reaches_strict_failure_ledger(retriever, monkeypatch):
    calls, _ = fake_responses(monkeypatch, [error(503), error(503), error(503)])
    asset = retriever.fetch_and_store(
        source_id="openfda_drug_label", compound="sildenafil", group="openfda_drug_label",
        request_key="label_skip_0", url=URL,
    )
    assert asset is None and len(calls) == 3
    failures = [json.loads(line) for line in retriever.failures_path.read_text().splitlines()]
    assert len(failures) == 1 and failures[0]["status"] == "failed"
    assert "503" in failures[0]["error"]


def test_404_is_not_retried_or_reclassified(retriever, monkeypatch):
    calls, delays = fake_responses(monkeypatch, [error(404)])
    with pytest.raises(HTTPError) as caught:
        retriever.fetch_bytes(URL)
    assert caught.value.code == 404
    assert calls == [URL] and delays == [] and events(retriever) == []


@pytest.mark.parametrize("header,expected", [
    ("999999", 60.0), ("3", 3.0), ("-9", 0.0), ("bad-header", 1.0),
    ("Sun, 11 Oct 2026 00:01:30 GMT", 60.0),
])
def test_retry_after_is_bounded(retriever, monkeypatch, header, expected):
    monkeypatch.setattr(retrieval, "utc_now", lambda: datetime(2026, 10, 11, tzinfo=timezone.utc))
    calls, delays = fake_responses(monkeypatch, [error(429, header), b"{}"])
    retriever.fetch_bytes(URL)
    assert len(calls) == 2 and delays == [expected]
    assert events(retriever)[0]["retry_delay_seconds"] == expected
