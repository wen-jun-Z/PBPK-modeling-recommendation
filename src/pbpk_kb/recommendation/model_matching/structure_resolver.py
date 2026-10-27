from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.model_matching.benchmark import load_model_records


RESOLVER_VERSION = "pbpk-pubchem-structure-resolver-1.2.0"
IDENTIFIER_ALIAS_VERSION = "pbpk-pubchem-identifier-aliases-1.0.0"
PUBCHEM_IDENTIFIER_ALIASES: dict[str, dict[str, str]] = {
    "n-desalkyl-itraconazole": {
        "identifier": "89848-41-9",
        "identifier_type": "CAS RN",
        "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/53789808",
    }
}


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _normalise_compound(value: object) -> str:
    return " ".join(str(value or "").replace("_", " ").split())


def _identifier_candidates(compound: str) -> list[dict[str, str | None]]:
    values: list[dict[str, str | None]] = [
        {
            "identifier": compound,
            "identifier_type": "compound_name",
            "alias_source_url": None,
        }
    ]
    alias = PUBCHEM_IDENTIFIER_ALIASES.get(compound.casefold())
    if alias is not None:
        values.append(
            {
                "identifier": alias["identifier"],
                "identifier_type": alias["identifier_type"],
                "alias_source_url": alias["source_url"],
            }
        )
    return values


def _benchmark_compounds(benchmark_directory: Path) -> tuple[list[str], dict[str, str]]:
    tasks_path = benchmark_directory / "benchmark_tasks.jsonl"
    candidates_path = benchmark_directory / "candidates.jsonl"
    if not tasks_path.is_file():
        raise FileNotFoundError(tasks_path)
    if not candidates_path.is_file():
        raise FileNotFoundError(candidates_path)
    compounds: dict[str, str] = {}

    def add(value: object) -> None:
        compound = _normalise_compound(value)
        if compound:
            compounds.setdefault(compound.casefold(), compound)

    for row in _iter_jsonl(tasks_path):
        for compound in row.get("required_model_compounds") or row.get(
            "active_moieties"
        ) or []:
            add(compound)
    for row in _iter_jsonl(candidates_path):
        for compound in row.get("active_moieties") or []:
            add(compound)
    return (
        sorted(compounds.values(), key=str.casefold),
        {
            "benchmark_tasks.jsonl": _sha256(tasks_path),
            "candidates.jsonl": _sha256(candidates_path),
        },
    )


def _model_family_compounds(model_paths: list[Path]) -> list[str]:
    records = load_model_records(model_paths)
    compounds = {
        _normalise_compound(row.get("model_family"))
        for row in records
        if _normalise_compound(row.get("model_family"))
    }
    return sorted(compounds, key=str.casefold)


def _reusable_structure_rows(path: Path | None) -> tuple[dict[str, dict[str, Any]], str | None]:
    if path is None:
        return {}, None
    if not path.is_file():
        raise FileNotFoundError(path)
    reusable: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        compound = _normalise_compound(row.get("compound"))
        raw_value = row.get("raw_response_path")
        if not compound or row.get("status") != "resolved" or not raw_value:
            continue
        raw_path = Path(str(raw_value))
        if not raw_path.is_absolute():
            raw_path = path.parent / raw_path
        expected = str(row.get("response_sha256") or "")
        if not raw_path.is_file() or not expected or _sha256(raw_path) != expected:
            continue
        reusable[compound.casefold()] = {**row, "_verified_raw_path": str(raw_path)}
    return reusable, _sha256(path)


def resolve_structures(
    *,
    output_directory: Path,
    model_paths: list[Path] | None = None,
    benchmark_directory: Path | None = None,
    existing_structures_path: Path | None = None,
    delay_seconds: float = 0.15,
) -> dict[str, Any]:
    if bool(model_paths) == bool(benchmark_directory):
        raise ValueError("Supply exactly one of model_paths or benchmark_directory.")
    output_directory.mkdir(parents=True, exist_ok=False)
    raw_directory = output_directory / "raw_responses"
    raw_directory.mkdir()
    benchmark_checksums: dict[str, str] = {}
    if benchmark_directory is not None:
        compounds, benchmark_checksums = _benchmark_compounds(benchmark_directory)
        input_mode = "execution_derived_benchmark_active_moieties"
    else:
        compounds = _model_family_compounds(model_paths or [])
        input_mode = "legacy_model_family"
    reusable, existing_checksum = _reusable_structure_rows(existing_structures_path)
    rows = []
    reused_count = 0
    downloaded_count = 0
    for compound in compounds:
        cached = reusable.get(compound.casefold())
        if cached is not None:
            raw_source = Path(cached.pop("_verified_raw_path"))
            raw_name = f"{hashlib.sha256(compound.casefold().encode()).hexdigest()[:20]}.json"
            raw_target = raw_directory / raw_name
            raw_checksum = _atomic_bytes(raw_target, raw_source.read_bytes())
            rows.append(
                {
                    **cached,
                    "compound": compound,
                    "raw_response_path": str(raw_target.relative_to(output_directory)),
                    "response_sha256": raw_checksum,
                    "cache_reused_from": str(existing_structures_path),
                    "cache_reused_at": datetime.now(UTC).isoformat(),
                }
            )
            reused_count += 1
            continue
        attempted_identifiers: list[dict[str, str | None]] = []
        failure: Exception | None = None
        resolved = False
        last_url = ""
        for query in _identifier_candidates(compound):
            identifier = str(query["identifier"])
            encoded = urllib.parse.quote(identifier, safe="")
            last_url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{encoded}/property/CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
            )
            attempted_identifiers.append(query)
            request = urllib.request.Request(
                last_url,
                headers={"User-Agent": "PBPK-Stage1-Public-KB/0.1"},
            )
            for attempt in range(1, 4):
                try:
                    with urllib.request.urlopen(request, timeout=30) as response:
                        raw = response.read()
                    value = json.loads(raw)
                    properties = value["PropertyTable"]["Properties"][0]
                    canonical = properties.get("ConnectivitySMILES") or properties.get(
                        "CanonicalSMILES"
                    )
                    raw_name = f"{hashlib.sha256(compound.casefold().encode()).hexdigest()[:20]}.json"
                    raw_path = raw_directory / raw_name
                    response_sha256 = _atomic_bytes(raw_path, raw)
                    rows.append(
                        {
                            "compound": compound,
                            "status": "resolved",
                            "pubchem_cid": properties.get("CID"),
                            "canonical_smiles": canonical,
                            "isomeric_smiles": properties.get("SMILES") or properties.get(
                                "IsomericSMILES"
                            ),
                            "inchi_key": properties.get("InChIKey"),
                            "source_id": "pubchem",
                            "source_url": last_url,
                            "query_identifier": identifier,
                            "query_identifier_type": query["identifier_type"],
                            "alias_source_url": query["alias_source_url"],
                            "response_sha256": response_sha256,
                            "raw_response_path": str(raw_path.relative_to(output_directory)),
                            "retrieved_at": datetime.now(UTC).isoformat(),
                        }
                    )
                    downloaded_count += 1
                    failure = None
                    resolved = True
                    break
                except urllib.error.HTTPError as exc:
                    failure = exc
                    if exc.code == 404:
                        break
                except (
                    urllib.error.URLError,
                    http.client.RemoteDisconnected,
                    TimeoutError,
                    ConnectionError,
                    KeyError,
                    json.JSONDecodeError,
                ) as exc:
                    failure = exc
                time.sleep(attempt)
            if resolved:
                break
        if failure is not None:
            rows.append(
                {
                    "compound": compound,
                    "status": "unresolved",
                    "canonical_smiles": None,
                    "source_id": "pubchem",
                    "source_url": last_url,
                    "attempted_identifiers": attempted_identifiers,
                    "failure": str(failure),
                    "retrieved_at": datetime.now(UTC).isoformat(),
                }
            )
        time.sleep(delay_seconds)
    checksum = _atomic_jsonl(output_directory / "compound_structures.jsonl", rows)
    resolved = sum(row["status"] == "resolved" and bool(row.get("canonical_smiles")) for row in rows)
    manifest = {
        "resolver_version": RESOLVER_VERSION,
        "source_id": "pubchem",
        "identifier_alias_version": IDENTIFIER_ALIAS_VERSION,
        "input_mode": input_mode,
        "compound_count": len(compounds),
        "model_family_count": len(compounds),
        "resolved_count": resolved,
        "unresolved_count": len(rows) - resolved,
        "coverage": resolved / len(rows) if rows else 0.0,
        "reused_count": reused_count,
        "downloaded_count": downloaded_count,
        "compound_structures_sha256": checksum,
        "input_model_paths": [str(path) for path in model_paths or []],
        "input_benchmark_directory": (
            str(benchmark_directory.resolve()) if benchmark_directory else None
        ),
        "input_benchmark_checksums": benchmark_checksums,
        "existing_structures_path": (
            str(existing_structures_path.resolve()) if existing_structures_path else None
        ),
        "existing_structures_sha256": existing_checksum,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve model-family chemical structures from PubChem with provenance."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--model-versions", action="append", type=Path)
    inputs.add_argument("--benchmark", type=Path)
    parser.add_argument("--existing-structures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = resolve_structures(
        model_paths=args.model_versions,
        benchmark_directory=args.benchmark,
        existing_structures_path=args.existing_structures,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
