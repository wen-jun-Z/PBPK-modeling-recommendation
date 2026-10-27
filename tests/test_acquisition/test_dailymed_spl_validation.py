import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_STORED, ZipFile

import pytest

from pbpk_kb.acquisition import comprehensive_raw_retriever as retrieval


SETID = "0720268c-2c92-4632-8c55-76c254d1beb8"
URL = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{SETID}.xml"
VALID = f'<document xmlns="urn:hl7-org:v3"><setId root="{SETID}"/><title>SILDENAFIL</title></document>'.encode()


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
    monkeypatch.setattr(instance, "fetch_dailymed_archive_bytes",
                        lambda url: pytest.fail("Archive network access must be mocked"))
    return instance


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def fake_payloads(monkeypatch, retriever, payloads):
    calls, delays = [], []
    outcomes = iter(payloads)

    def fetch(url):
        calls.append(url)
        return next(outcomes), None, 200, "curl fallback"

    monkeypatch.setattr(retriever, "fetch_bytes", fetch)
    monkeypatch.setattr(retrieval.time, "sleep", delays.append)
    return calls, delays


def request(retriever, version=2):
    return retriever.fetch_and_store(
        source_id="dailymed", compound="sildenafil", group="dailymed/spl_xml",
        request_key=f"spl_xml_{SETID}", url=URL, expected_extension="xml",
        metadata={"setid": SETID, "title": "SILDENAFIL", "spl_version": version},
    )


@pytest.mark.parametrize("payload", [
    b"None", b"", b"<html>service unavailable</html>",
    b"<document><setId root='0720268c-2c92-4632-8c55-76c254d1beb8'/></document>",
    b'<document xmlns="urn:hl7-org:v3"><setId root="wrong"/></document>',
    b'<document xmlns="urn:hl7-org:v3"/>',
    f'<document xmlns="urn:hl7-org:v3"><component><setId root="{SETID}"/></component></document>'.encode(),
])
def test_invalid_success_retries_then_registers_only_valid_document(retriever, monkeypatch, payload):
    calls, delays = fake_payloads(monkeypatch, retriever, [payload, VALID])
    asset = request(retriever)
    assert asset is not None
    assert calls == [URL, URL] and delays == [1.0]
    assert Path(asset["storage_path"]).read_bytes() == VALID
    assert len(rows(retriever.assets_path)) == 1 and not rows(retriever.failures_path)
    assert [row["status"] for row in rows(retriever.requests_path)] == ["started", "fetched"]
    attempt = rows(retriever.run_root / "transport_attempts.jsonl")[0]
    assert attempt["failure_type"] == "dailymed_spl_content_invalid"
    assert attempt["byte_count"] == len(payload) and attempt["checksum"]


def test_repeated_invalid_payload_is_bounded_and_still_fails_strictly(retriever, monkeypatch):
    calls, delays = fake_payloads(monkeypatch, retriever, [b"None"] * 3)
    assert request(retriever, version=None) is None
    assert calls == [URL] * 3 and delays == [1.0, 2.0]
    assert rows(retriever.assets_path) == []
    assert [row["status"] for row in rows(retriever.requests_path)] == ["started", "failed"]
    assert len(rows(retriever.failures_path)) == 1
    attempts = rows(retriever.run_root / "transport_attempts.jsonl")
    assert [row["attempt"] for row in attempts] == [1, 2, 3]
    assert attempts[-1]["status"] == "retry_exhausted"


def test_valid_label_is_accepted_without_retry(retriever, monkeypatch):
    calls, delays = fake_payloads(monkeypatch, retriever, [VALID])
    assert request(retriever) is not None
    assert calls == [URL] and delays == []
    assert rows(retriever.run_root / "transport_attempts.jsonl") == []


@pytest.mark.parametrize("url,setid", [
    (URL.replace(SETID, "00000000-0000-0000-0000-000000000000"), SETID),
    (URL.replace("dailymed.nlm.nih.gov", "example.invalid"), SETID),
    (URL + "?another_label=1", SETID), (URL, ""),
])
def test_document_identity_cannot_validate_another_request(url, setid):
    with pytest.raises(ValueError, match="request identity"):
        retrieval.ComprehensiveRawRetriever.validate_dailymed_spl(VALID, url, setid)


def test_other_daily_med_requests_are_not_forced_to_be_spl(retriever, monkeypatch):
    calls, delays = fake_payloads(monkeypatch, retriever, [b'{"data": []}'])
    asset = retriever.fetch_and_store(source_id="dailymed", compound="sildenafil", group="dailymed",
                                     request_key="listing", url=URL.rsplit("/", 1)[0] + ".json")
    assert asset is not None and len(calls) == 1 and delays == []


def test_resume_cache_cannot_bypass_content_validation(retriever, tmp_path, monkeypatch):
    cached_path = tmp_path / "bad.xml"
    cached_path.write_bytes(b"None")
    key = retriever.request_id("dailymed", "sildenafil", f"spl_xml_{SETID}", URL, None)
    retriever._resume_assets = {key: {"storage_path": str(cached_path)}}
    monkeypatch.setattr(retriever, "fetch_bytes", lambda url: pytest.fail("Cached invalid bytes need explicit recovery"))
    assert request(retriever) is None
    assert "Cached DailyMed SPL failed validation" in rows(retriever.failures_path)[0]["error"]


def spl_zip(*, setid=SETID, version=2, member="label.xml", duplicate=False):
    xml = (f'<document xmlns="urn:hl7-org:v3"><id root="11111111-2222-3333-4444-555555555555"/>'
           f'<setId root="{setid}"/><versionNumber value="{version}"/><title>SILDENAFIL</title></document>').encode()
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_STORED) as archive:
        archive.writestr(member, xml)
        archive.writestr("picture.jpg", b"picture bytes")
        if duplicate:
            archive.writestr("second.xml", xml)
    return stream.getvalue()


def archive_outcome(monkeypatch, retriever, payload, status=200):
    calls = []

    def fetch(url):
        calls.append(url)
        return payload, "application/zip", status, None

    monkeypatch.setattr(retriever, "fetch_dailymed_archive_bytes", fetch)
    return calls


def test_content_failure_recovers_one_verified_exact_version_archive(retriever, monkeypatch):
    calls, delays = fake_payloads(monkeypatch, retriever, [b"None"] * 3)
    payload = spl_zip()
    archive_calls = archive_outcome(monkeypatch, retriever, payload)
    asset = request(retriever)
    assert asset is not None
    assert calls == [URL] * 3 and delays == [1.0, 2.0]
    assert archive_calls == [f"https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?type=zip&setid={SETID}&version=2"]
    assert asset["url"] == archive_calls[0] and asset["group"] == "dailymed/spl_archive"
    assert Path(asset["storage_path"]).read_bytes() == payload
    assert asset["metadata"]["original_xml_url"] == URL
    assert asset["metadata"]["original_xml_request_id"] == retriever.request_id("dailymed", "sildenafil", f"spl_xml_{SETID}", URL, None)
    assert asset["metadata"]["setid"] == SETID and asset["metadata"]["spl_version"] == 2
    assert asset["metadata"]["member_sha256"] and asset["metadata"]["archive_member_count"] == 2
    assert len(rows(retriever.assets_path)) == 1 and not rows(retriever.failures_path)
    requests = rows(retriever.requests_path)
    assert [row["status"] for row in requests] == ["started", "started", "fetched", "recovered_by_verified_archive"]
    assert requests[2]["url"] == archive_calls[0] and requests[3]["url"] == URL
    assert requests[3]["asset_id"] == asset["asset_id"]
    assert rows(retriever.run_root / "transport_attempts.jsonl")[-1]["status"] == "retry_exhausted"


@pytest.mark.parametrize("payload", [
    b"None", b"not a zip", spl_zip(version=3),
    spl_zip(setid="00000000-0000-0000-0000-000000000000"),
    spl_zip(member="../label.xml"), spl_zip(member="/label.xml"), spl_zip(duplicate=True),
])
def test_bad_archive_does_not_resolve_the_original_xml_failure(retriever, monkeypatch, payload):
    fake_payloads(monkeypatch, retriever, [b"None"] * 3)
    archive_calls = archive_outcome(monkeypatch, retriever, payload)
    assert request(retriever) is None
    assert len(archive_calls) == 1 and not rows(retriever.assets_path)
    failures = rows(retriever.failures_path)
    assert len(failures) == 2
    assert {row["url"] for row in failures} == {URL, archive_calls[0]}
    assert "recovered_by_verified_archive" not in {row["status"] for row in rows(retriever.requests_path)}


@pytest.mark.parametrize("version", [None, 0, -1, True, "2", 2.5])
def test_archive_fallback_requires_known_positive_integer_version(retriever, monkeypatch, version):
    fake_payloads(monkeypatch, retriever, [b"None"] * 3)
    assert request(retriever, version=version) is None
    assert len(rows(retriever.failures_path)) == 1


def test_archive_crc_is_checked_before_asset_registration(retriever, monkeypatch):
    payload = bytearray(spl_zip())
    with ZipFile(BytesIO(payload)) as archive:
        info = archive.getinfo("picture.jpg")
        offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    payload[offset] ^= 1
    fake_payloads(monkeypatch, retriever, [b"None"] * 3)
    archive_outcome(monkeypatch, retriever, bytes(payload))
    assert request(retriever) is None
    assert not rows(retriever.assets_path)
    assert any("CRC" in row["error"] for row in rows(retriever.failures_path))


@pytest.mark.parametrize("status", [204, 301, 500])
def test_archive_non_200_response_never_counts_as_recovery(retriever, monkeypatch, status):
    fake_payloads(monkeypatch, retriever, [b"None"] * 3)
    archive_outcome(monkeypatch, retriever, spl_zip(), status=status)
    assert request(retriever) is None and not rows(retriever.assets_path)


def test_archive_byte_limit_is_enforced_before_validation(retriever, monkeypatch):
    from pbpk_kb.acquisition import dailymed_recovery
    monkeypatch.setattr(dailymed_recovery, "MAX_ARCHIVE_BYTES", 5)
    fake_payloads(monkeypatch, retriever, [b"None"] * 3)
    archive_outcome(monkeypatch, retriever, spl_zip())
    assert request(retriever) is None and not rows(retriever.assets_path)


@pytest.mark.parametrize("http,effective_suffix", [("200", ""), ("302", ""), ("200", "&wrong=1")])
def test_archive_transport_is_bounded_exact_and_cleans_temporary_download(retriever, monkeypatch, http, effective_suffix):
    url = f"https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?type=zip&setid={SETID}&version=2"
    calls = []
    archive_payload = spl_zip()

    def curl(command, **kwargs):
        calls.append(command)
        assert command[command.index("--retry") + 1] == "0"
        assert command[command.index("--max-filesize") + 1] == str(32 * 1024 * 1024)
        assert "--insecure" not in command and command[-1] == url
        target = Path(command[command.index("--output") + 1])
        assert target.is_relative_to(retriever.run_root)
        target.write_bytes(archive_payload)
        return SimpleNamespace(stdout=f"{http}\n{url}{effective_suffix}\napplication/zip\n".encode())

    monkeypatch.setattr(retrieval.subprocess, "run", curl)
    method = retrieval.ComprehensiveRawRetriever.fetch_dailymed_archive_bytes
    if http == "200" and not effective_suffix:
        payload, content_type, status, warning = method(retriever, url)
        assert payload == archive_payload and content_type == "application/zip" and status == 200
    else:
        with pytest.raises(ValueError, match="exact official URL"):
            method(retriever, url)
    assert len(calls) == 1
    assert list(retriever.run_root.glob(".dailymed-archive-*")) == []
