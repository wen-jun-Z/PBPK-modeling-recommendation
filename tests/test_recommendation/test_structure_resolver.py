from __future__ import annotations

import http.client
import json
import urllib.error
from pathlib import Path

import pytest

from pbpk_kb.recommendation.model_matching.structure_resolver import resolve_structures


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _pubchem_payload() -> bytes:
    return json.dumps(
        {
            "PropertyTable": {
                "Properties": [
                    {
                        "CID": 123,
                        "ConnectivitySMILES": "CCO",
                        "SMILES": "CCO",
                        "InChIKey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                    }
                ]
            }
        }
    ).encode()


def test_resolver_uses_task_and_candidate_active_moiety_union(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    _write_jsonl(
        benchmark / "benchmark_tasks.jsonl",
        [
            {
                "required_model_compounds": ["Drug A", "Metabolite B"],
                "active_moieties": ["ignored fallback"],
            },
            {"active_moieties": ["Drug C"]},
        ],
    )
    _write_jsonl(
        benchmark / "candidates.jsonl",
        [{"model_family": "Drug A+Drug D", "active_moieties": ["Drug A", "Drug D"]}],
    )
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: _Response(_pubchem_payload())
    )

    output = tmp_path / "structures"
    manifest = resolve_structures(
        benchmark_directory=benchmark,
        output_directory=output,
        delay_seconds=0,
    )

    rows = [json.loads(line) for line in (output / "compound_structures.jsonl").read_text().splitlines()]
    assert [row["compound"] for row in rows] == ["Drug A", "Drug C", "Drug D", "Metabolite B"]
    assert manifest["input_mode"] == "execution_derived_benchmark_active_moieties"
    assert manifest["compound_count"] == 4
    assert manifest["resolved_count"] == 4
    assert manifest["coverage"] == 1.0
    assert manifest["downloaded_count"] == 4
    for row in rows:
        raw_path = output / row["raw_response_path"]
        assert raw_path.is_file()
        assert row["response_sha256"]


def test_resolver_reuses_only_checksum_verified_raw_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    _write_jsonl(benchmark / "benchmark_tasks.jsonl", [{"active_moieties": ["Drug A"]}])
    _write_jsonl(benchmark / "candidates.jsonl", [{"active_moieties": ["Drug A"]}])
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: _Response(_pubchem_payload())
    )
    first = tmp_path / "first"
    resolve_structures(
        benchmark_directory=benchmark,
        output_directory=first,
        delay_seconds=0,
    )

    def unexpected_download(*args: object, **kwargs: object) -> object:
        raise AssertionError("verified cache should prevent a download")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_download)
    second = tmp_path / "second"
    manifest = resolve_structures(
        benchmark_directory=benchmark,
        existing_structures_path=first / "compound_structures.jsonl",
        output_directory=second,
        delay_seconds=0,
    )

    assert manifest["reused_count"] == 1
    assert manifest["downloaded_count"] == 0
    row = json.loads((second / "compound_structures.jsonl").read_text().splitlines()[0])
    assert row["cache_reused_from"] == str(first / "compound_structures.jsonl")
    assert (second / row["raw_response_path"]).is_file()


def test_resolver_retries_remote_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    _write_jsonl(benchmark / "benchmark_tasks.jsonl", [{"active_moieties": ["Drug A"]}])
    _write_jsonl(benchmark / "candidates.jsonl", [{"active_moieties": ["Drug A"]}])
    attempts = 0

    def transient_disconnect(*args: object, **kwargs: object) -> _Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise http.client.RemoteDisconnected("closed")
        return _Response(_pubchem_payload())

    monkeypatch.setattr("urllib.request.urlopen", transient_disconnect)
    manifest = resolve_structures(
        benchmark_directory=benchmark,
        output_directory=tmp_path / "structures",
        delay_seconds=0,
    )

    assert attempts == 2
    assert manifest["resolved_count"] == 1


def test_resolver_uses_provenance_locked_identifier_alias_after_name_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    compound = "n-desalkyl-itraconazole"
    _write_jsonl(benchmark / "benchmark_tasks.jsonl", [{"active_moieties": [compound]}])
    _write_jsonl(benchmark / "candidates.jsonl", [{"active_moieties": [compound]}])
    requested_urls: list[str] = []

    def resolve_alias(request: object, *args: object, **kwargs: object) -> _Response:
        url = str(getattr(request, "full_url"))
        requested_urls.append(url)
        if "n-desalkyl-itraconazole" in url:
            raise urllib.error.HTTPError(url, 404, "not found", None, None)
        return _Response(_pubchem_payload())

    monkeypatch.setattr("urllib.request.urlopen", resolve_alias)
    output = tmp_path / "structures"
    manifest = resolve_structures(
        benchmark_directory=benchmark,
        output_directory=output,
        delay_seconds=0,
    )

    row = json.loads((output / "compound_structures.jsonl").read_text().splitlines()[0])
    assert requested_urls[-1].endswith(
        "/89848-41-9/property/CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
    )
    assert row["query_identifier"] == "89848-41-9"
    assert row["query_identifier_type"] == "CAS RN"
    assert row["alias_source_url"] == "https://pubchem.ncbi.nlm.nih.gov/compound/53789808"
    assert manifest["identifier_alias_version"]


def test_resolver_requires_exactly_one_input_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        resolve_structures(output_directory=tmp_path / "none", delay_seconds=0)
