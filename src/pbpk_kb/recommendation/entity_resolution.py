from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import httpx

from pbpk_kb.recommendation.entity_extraction import normalize_text
from pbpk_kb.recommendation.kg_models import stable_id, utc_now


PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
RESOLVER_VERSION = "public-identity-resolver-1.0.0"
GENERIC_TERMS = {
    "active ingredient",
    "agent",
    "c",
    "cc",
    "chemical",
    "compound",
    "control",
    "drug",
    "drugs",
    "food",
    "inhibitor",
    "medication",
    "metabolite",
    "o",
    "placebo",
    "product",
    "substrate",
    "treatment",
    "unchanged",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve external DDI compounds through PubChem and RxNorm.")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--delay-seconds", type=float, default=0.22)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    summary = run_resolution(
        evidence_path=args.evidence,
        unresolved_path=args.unresolved,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
        delay_seconds=args.delay_seconds,
        reset=args.reset,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def run_resolution(
    *,
    evidence_path: Path,
    unresolved_path: Path,
    output_dir: Path,
    min_confidence: float,
    delay_seconds: float,
    reset: bool,
) -> dict[str, Any]:
    if reset and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "resolver_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    evidence = list(_iter_jsonl(evidence_path))
    unresolved = list(_iter_jsonl(unresolved_path))
    candidates = collect_candidates(evidence, unresolved, min_confidence=min_confidence)
    client = TraceableResolverClient(raw_dir=raw_dir, delay_seconds=delay_seconds)
    decisions = []
    catalog: dict[str, dict[str, Any]] = {}
    try:
        for index, candidate in enumerate(candidates, start=1):
            decision = resolve_candidate(candidate, client)
            decisions.append(decision)
            if decision["status"] == "resolved":
                entry = decision["catalog_entry"]
                catalog[entry["node_id"]] = entry
            if index % 10 == 0 or index == len(candidates):
                print(json.dumps({"resolved_candidates": index, "total": len(candidates)}), flush=True)
    finally:
        client.close()

    status_counts = Counter(item["status"] for item in decisions)
    method_counts = Counter(item.get("resolution_method") for item in decisions)
    summary = {
        "accepted": len(decisions) == len(candidates),
        "resolver_version": RESOLVER_VERSION,
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
        "unresolved_path": str(unresolved_path),
        "unresolved_sha256": _sha256(unresolved_path),
        "candidate_count": len(candidates),
        "resolved_count": len(catalog),
        "status_counts": dict(sorted(status_counts.items())),
        "method_counts": dict(sorted((str(key), value) for key, value in method_counts.items())),
        "raw_response_count": client.response_count,
        "generated_at": utc_now(),
    }
    _atomic_jsonl(output_dir / "external_entity_catalog.jsonl", sorted(catalog.values(), key=lambda x: x["node_id"]))
    _atomic_jsonl(output_dir / "entity_resolution_decisions.jsonl", decisions)
    _atomic_json(output_dir / "entity_resolution_summary.json", summary)
    return summary


def collect_candidates(
    evidence: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    *,
    min_confidence: float,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    evidence_by_id = {str(row["evidence_id"]): row for row in evidence}
    for item in unresolved:
        if item.get("entity_type") != "Compound":
            continue
        if float(item.get("recognition_confidence") or 0.0) < min_confidence:
            continue
        evidence_row = evidence_by_id.get(str(item.get("evidence_id")))
        if not evidence_row or evidence_row.get("evidence_type") != "ddi":
            continue
        _add_candidate(
            candidates,
            str(item.get("mention_text") or ""),
            evidence_id=str(item.get("evidence_id")),
            source="ddi_ner_mention",
        )
    for row in evidence:
        if row.get("evidence_type") != "ddi":
            continue
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        for role in ("victim_drug", "perpetrator_drug", "interacting_drug"):
            for value in _recursive_values(metadata, role):
                _add_candidate(
                    candidates,
                    value,
                    evidence_id=str(row["evidence_id"]),
                    source=f"structured_{role}",
                    role=role,
                )
    return sorted(candidates.values(), key=lambda item: item["normalized_name"])


def _add_candidate(
    candidates: dict[str, dict[str, Any]],
    name: str,
    *,
    evidence_id: str,
    source: str,
    role: str | None = None,
) -> None:
    cleaned = " ".join(str(name or "").split())
    normalized = normalize_text(cleaned)
    if not _eligible_name(normalized):
        return
    item = candidates.setdefault(
        normalized,
        {
            "name": cleaned,
            "normalized_name": normalized,
            "evidence_ids": [],
            "sources": [],
            "roles": [],
        },
    )
    if evidence_id not in item["evidence_ids"]:
        item["evidence_ids"].append(evidence_id)
    if source not in item["sources"]:
        item["sources"].append(source)
    if role and role not in item["roles"]:
        item["roles"].append(role)


def _eligible_name(normalized: str) -> bool:
    if normalized in GENERIC_TERMS or len(normalized) < 3 or len(normalized) > 100:
        return False
    if not re.search(r"[a-z]", normalized):
        return False
    if len(normalized.split()) > 8:
        return False
    return True


def resolve_candidate(candidate: dict[str, Any], client: "TraceableResolverClient") -> dict[str, Any]:
    name = candidate["name"]
    query_name = _canonicalize_identifier_punctuation(name)
    candidate = {**candidate, "resolution_query_variant": query_name}
    pubchem = resolve_pubchem(query_name, client)
    rxnorm = resolve_rxnorm(query_name, client)
    resolved_ids = {}
    preferred_name = name
    aliases = {name}
    if pubchem["status"] == "resolved":
        resolved_ids["PUBCHEM.COMPOUND"] = str(pubchem["cid"])
        resolved_ids["INCHIKEY"] = str(pubchem.get("inchi_key") or "")
        preferred_name = str(pubchem.get("title") or name)
        aliases.update(pubchem.get("matching_synonyms") or [])
    if rxnorm["status"] == "resolved":
        resolved_ids["RXNORM"] = str(rxnorm["rxcui"])
        aliases.add(str(rxnorm.get("name") or name))
    resolved_ids = {key: value for key, value in resolved_ids.items() if value}
    if pubchem["status"] == "ambiguous" or rxnorm["status"] == "ambiguous":
        return {
            **candidate,
            "status": "ambiguous",
            "resolution_method": "pubchem_rxnorm_exact",
            "pubchem": pubchem,
            "rxnorm": rxnorm,
            "reasons": ["at least one authoritative resolver returned multiple exact concepts"],
        }
    if not resolved_ids:
        return {
            **candidate,
            "status": "unresolved",
            "resolution_method": "pubchem_rxnorm_exact",
            "pubchem": pubchem,
            "rxnorm": rxnorm,
            "reasons": sorted(
                set((pubchem.get("reason") or "pubchem_unresolved", rxnorm.get("reason") or "rxnorm_unresolved"))
            ),
        }
    primary = resolved_ids.get("PUBCHEM.COMPOUND") or f"RXNORM:{resolved_ids['RXNORM']}"
    entry = {
        "node_id": f"Compound:PUBLIC:{primary}",
        "node_type": "Compound",
        "preferred_name": preferred_name,
        "aliases": sorted(aliases, key=str.lower),
        "standard_identifiers": resolved_ids,
        "resolution_method": "pubchem_rxnorm_exact",
        "resolution_query": name,
        "resolution_query_variant": query_name,
    }
    return {
        **candidate,
        "status": "resolved",
        "resolution_method": "pubchem_rxnorm_exact",
        "pubchem": pubchem,
        "rxnorm": rxnorm,
        "catalog_entry": entry,
        "reasons": [],
    }


def _canonicalize_identifier_punctuation(value: str) -> str:
    return str(value).translate(
        {
            ord("\u2010"): "-",
            ord("\u2011"): "-",
            ord("\u2012"): "-",
            ord("\u2013"): "-",
            ord("\u2014"): "-",
            ord("\u2212"): "-",
        }
    )


def resolve_pubchem(name: str, client: "TraceableResolverClient") -> dict[str, Any]:
    encoded = quote(name, safe="")
    cids_response = client.get_json(f"{PUBCHEM_BASE}/compound/name/{encoded}/cids/JSON", "pubchem_cids")
    cids = (
        cids_response.get("IdentifierList", {}).get("CID", [])
        if isinstance(cids_response, dict)
        else []
    )
    exact = []
    for cid in list(dict.fromkeys(str(value) for value in cids))[:10]:
        synonyms_response = client.get_json(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON",
            "pubchem_synonyms",
        )
        synonyms = _pubchem_synonyms(synonyms_response)
        matching = [value for value in synonyms if normalize_text(value) == normalize_text(name)]
        if not matching:
            continue
        properties_response = client.get_json(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/Title,InChIKey/JSON",
            "pubchem_properties",
        )
        properties = _first_pubchem_property(properties_response)
        exact.append(
            {
                "cid": cid,
                "title": properties.get("Title"),
                "inchi_key": properties.get("InChIKey"),
                "matching_synonyms": matching[:10],
            }
        )
    if len(exact) == 1:
        return {"status": "resolved", **exact[0]}
    if len(exact) > 1:
        return {"status": "ambiguous", "matches": exact, "reason": "multiple_exact_pubchem_cids"}
    return {"status": "unresolved", "reason": "no_exact_pubchem_synonym"}


def resolve_rxnorm(name: str, client: "TraceableResolverClient") -> dict[str, Any]:
    encoded = quote(name, safe="")
    response = client.get_json(f"{RXNORM_BASE}/rxcui.json?name={encoded}&search=2", "rxnorm_exact")
    ids = (
        response.get("idGroup", {}).get("rxnormId", [])
        if isinstance(response, dict)
        else []
    ) or []
    ids = list(dict.fromkeys(str(value) for value in ids))
    if len(ids) == 1:
        properties = client.get_json(f"{RXNORM_BASE}/rxcui/{ids[0]}/properties.json", "rxnorm_properties")
        name_value = (
            properties.get("properties", {}).get("name")
            if isinstance(properties, dict)
            else None
        )
        if normalize_text(name_value) == normalize_text(name):
            return {"status": "resolved", "rxcui": ids[0], "name": name_value}
        return {"status": "unresolved", "reason": "rxnorm_name_not_exact"}
    if len(ids) > 1:
        return {"status": "ambiguous", "rxcuis": ids, "reason": "multiple_exact_rxnorm_ids"}
    return {"status": "unresolved", "reason": "no_exact_rxnorm_concept"}


class TraceableResolverClient:
    def __init__(self, *, raw_dir: Path, delay_seconds: float) -> None:
        self.raw_dir = raw_dir
        self.delay_seconds = delay_seconds
        self.client = httpx.Client(
            timeout=httpx.Timeout(45.0),
            headers={"User-Agent": "pbpk-public-kb/1.0 (public biomedical entity resolution)"},
            follow_redirects=True,
        )
        self.response_count = 0

    def get_json(self, url: str, kind: str) -> dict[str, Any]:
        request_id = stable_id("resolver-request", url)
        payload_path = self.raw_dir / f"{request_id.replace(':', '_')}.json"
        metadata_path = payload_path.with_suffix(".metadata.json")
        if payload_path.exists() and metadata_path.exists():
            return json.loads(payload_path.read_text(encoding="utf-8"))
        response: httpx.Response | None = None
        error = None
        for attempt in range(4):
            try:
                response = self.client.get(url)
                if response.status_code == 404:
                    data: dict[str, Any] = {}
                    break
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.HTTPError, ValueError) as exc:
                error = f"{type(exc).__name__}:{exc}"
                if attempt == 3:
                    data = {}
                    break
                time.sleep(1.5 * (attempt + 1))
        payload = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        _atomic_bytes(payload_path, payload)
        _atomic_json(
            metadata_path,
            {
                "request_id": request_id,
                "kind": kind,
                "url": url,
                "downloaded_at": utc_now(),
                "status_code": response.status_code if response is not None else None,
                "checksum": hashlib.sha256(payload).hexdigest(),
                "error": error,
                "resolver_version": RESOLVER_VERSION,
            },
        )
        self.response_count += 1
        time.sleep(self.delay_seconds)
        return data

    def close(self) -> None:
        self.client.close()


def _recursive_values(value: Any, wanted_key: str) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() == wanted_key.lower():
                if isinstance(item, str) and item.strip():
                    yield item
                elif isinstance(item, list):
                    for list_item in item:
                        if isinstance(list_item, str) and list_item.strip():
                            yield list_item
            yield from _recursive_values(item, wanted_key)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_values(item, wanted_key)


def _pubchem_synonyms(value: dict[str, Any]) -> list[str]:
    information = value.get("InformationList", {}).get("Information", []) if isinstance(value, dict) else []
    if not information:
        return []
    return [str(item) for item in information[0].get("Synonym", [])]


def _first_pubchem_property(value: dict[str, Any]) -> dict[str, Any]:
    properties = value.get("PropertyTable", {}).get("Properties", []) if isinstance(value, dict) else []
    return properties[0] if properties else {}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
