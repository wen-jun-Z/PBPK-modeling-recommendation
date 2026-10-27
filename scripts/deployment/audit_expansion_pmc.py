"""Audit actual PMC bytes and parsed bodies, rather than initial download caps.

Reads only the seven registered October runs. Original ledgers stay immutable.
An XML request or an article abstract is never counted as a full-text body.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

from defusedxml import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
RUNS = tuple(f"run_20261009T1409{i:02d}Z" for i in range(19, 26))
OUTPUT = ROOT / "runtime/deployment/evidence_expansion_20261009/recovery_quality_20261014/pmc_content_audit_20261026.json"


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def local(tag):
    return tag.rsplit("}", 1)[-1]


def pmcid(value):
    value = str(value or "").upper()
    if re.fullmatch(r"(?:PMC)?\d+", value):
        return "PMC" + value.removeprefix("PMC")
    return None


def article_content(payload):
    """IDs must be from the article's own front, never cited references."""
    root = ET.fromstring(payload)
    articles = [node for node in root.iter() if local(node.tag) == "article"]
    result = {}
    for article in articles:
        front = next((x for x in article if local(x.tag) == "front"), None)
        if front is None:
            continue
        meta = next((x for x in front if local(x.tag) == "article-meta"), None)
        if meta is None:
            continue
        ids = {pmcid(x.text) for x in meta if local(x.tag) == "article-id"
               and x.get("pub-id-type") in {"pmc", "pmcid"}}
        ids.discard(None)
        if len(ids) != 1:
            continue
        identity = next(iter(ids))
        body = next((x for x in article if local(x.tag) == "body"), None)
        body_chars = len(" ".join(body.itertext()).strip()) if body is not None else 0
        def narrative_paragraphs(node):
            if local(node.tag) in {"ref-list", "table-wrap", "fig", "supplementary-material", "ack", "fn-group"}:
                return 0
            return int(local(node.tag) == "p" and bool(" ".join(node.itertext()).strip())) + sum(
                narrative_paragraphs(child) for child in node)
        body_paragraphs = narrative_paragraphs(body) if body is not None else 0
        body_tables = sum(local(x.tag) == "table" for x in body.iter()) if body is not None else 0
        abstract_chars = sum(len(" ".join(x.itertext()).strip()) for x in meta
                             if local(x.tag) == "abstract")
        # A duplicate article in a response must not erase a fuller copy.
        previous = result.get(identity, {})
        result[identity] = {"body_chars": max(body_chars, previous.get("body_chars", 0)),
                            "body_paragraphs": max(body_paragraphs, previous.get("body_paragraphs", 0)),
                            "body_tables": max(body_tables, previous.get("body_tables", 0)),
                            "abstract_chars": max(abstract_chars, previous.get("abstract_chars", 0))}
    return result


def checked_asset(asset):
    path = Path(asset["storage_path"]).resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError("PMC asset escaped the project")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise RuntimeError("PMC XML exceeds bounded audit size")
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != asset["checksum"] or len(payload) != asset["byte_count"]:
        raise RuntimeError("PMC original checksum or size mismatch")
    return payload


def audit_run(run):
    acquisition = ROOT / "artifacts/pbpk_raw_runs/compound_runs" / run
    ledger = acquisition / "assets.jsonl"
    compounds_by_article = defaultdict(set)
    records = {}
    fingerprints = {str(ledger.relative_to(ROOT)): sha256(ledger.read_bytes()).hexdigest()}
    for path in sorted((acquisition / "compounds").glob("*/pmc_oa/pmc_fulltext_gap_manifest.json")):
        source = json.loads(path.read_text())
        fingerprints[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
        deferred = set(source["skipped_pmcids"])
        discovered = set(source["selected_for_fulltext_download"]) | deferred
        for identity in sorted(discovered):
            compound = source["compound"]
            compounds_by_article[identity].add(compound)
            records[compound, identity] = {"compound": compound, "pmcid": identity,
                "initially_deferred": identity in deferred, "xml_articles": [],
                "raw_body_available": False, "raw_abstract_available": False,
                "raw_body_paragraphs": 0, "raw_body_tables": 0, "parsed_tables": 0,
                "parsed_body_paragraphs": 0, "parsed_abstracts": 0}
    asset_index = {}
    checked = 0
    parse_errors = []
    nonarticle_metadata = []
    for asset in rows(ledger):
        if asset.get("source_id") != "pmc_oa":
            continue
        group = asset.get("group", "")
        if not (group.startswith("pmc/efetch_xml") or (
                group.startswith(("pmc/oa_s3_objects/", "pmc/oa_s3_files/"))
                and Path(asset["storage_path"]).suffix.lower() in {".xml", ".nxml"})):
            continue
        payload = checked_asset(asset)
        asset_index[asset["asset_id"]] = asset["checksum"]
        checked += 1
        try:
            content = article_content(payload)
        except ET.ParseError as exc:
            from pbpk_kb.parsers.raw_formats import recognize_publisher_data_availability
            if recognize_publisher_data_availability(payload) is not None:
                nonarticle_metadata.append({"asset_id": asset["asset_id"], "sha256": asset["checksum"],
                                            "status": "publisher_data_availability_metadata"})
                continue
            parse_errors.append({"asset_id": asset["asset_id"], "path": str(Path(asset["storage_path"]).relative_to(ROOT)),
                                 "error": str(exc), "sha256": asset["checksum"]})
            continue
        for identity, item in content.items():
            associations = ({asset["compound"]} if asset.get("compound")
                            else compounds_by_article.get(identity, set()))
            for compound in associations:
                row = records.get((compound, identity))
                if row is None:
                    continue
                row["xml_articles"].append({"asset_id": asset["asset_id"], "sha256": asset["checksum"],
                    "path": str(Path(asset["storage_path"]).relative_to(ROOT)), **item})
                row["raw_body_available"] |= item["body_chars"] > 0
                row["raw_body_paragraphs"] = max(row["raw_body_paragraphs"], item["body_paragraphs"])
                row["raw_body_tables"] = max(row["raw_body_tables"], item["body_tables"])
                row["raw_abstract_available"] |= item["abstract_chars"] > 0
    documents = ROOT / "data_process/raw_processing" / run / "02_parsed_outputs/parsed_documents.jsonl"
    digest = sha256()
    with documents.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            doc = json.loads(line)
            if doc.get("source_id") != "pmc_oa" or doc.get("content_kind") not in {"body_paragraph", "abstract"}:
                continue
            identity = pmcid((doc.get("metadata_json") or {}).get("pmcid"))
            if identity is None:
                continue
            if doc.get("raw_asset_id") not in asset_index:
                continue
            if doc["checksum"] != asset_index[doc["raw_asset_id"]]:
                raise RuntimeError("Parsed PMC paragraph checksum differs from its original")
            associations = {doc["compound"]} if doc.get("compound") else compounds_by_article.get(identity, set())
            for compound in associations:
                row = records.get((compound, identity))
                if row is not None:
                    field = "parsed_body_paragraphs" if doc["content_kind"] == "body_paragraph" else "parsed_abstracts"
                    row[field] += 1
    fingerprints[str(documents.relative_to(ROOT))] = digest.hexdigest()
    tables = documents.with_name("parsed_tables.jsonl")
    digest = sha256()
    with tables.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            table = json.loads(line)
            if table.get("source_id") != "pmc_oa" or table.get("raw_asset_id") not in asset_index:
                continue
            if table["checksum"] != asset_index[table["raw_asset_id"]]:
                raise RuntimeError("Parsed PMC table checksum differs from original")
            identity = pmcid((table.get("metadata_json") or {}).get("pmcid"))
            associations = {table["compound"]} if table.get("compound") else compounds_by_article.get(identity, set())
            for compound in associations:
                row = records.get((compound, identity))
                if row is not None:
                    row["parsed_tables"] += 1
    fingerprints[str(tables.relative_to(ROOT))] = digest.hexdigest()
    totals = Counter()
    deferred_totals = Counter()
    for row in records.values():
        row["status"] = ("body_parsed" if row["raw_body_available"] and row["parsed_body_paragraphs"]
            else "body_parse_gap" if row["raw_body_paragraphs"]
            else "table_only_parsed" if row["raw_body_tables"] and row["parsed_tables"]
            else "table_parse_gap" if row["raw_body_tables"]
            else "body_without_paragraphs" if row["raw_body_available"]
            else "abstract_only" if row["raw_abstract_available"]
            else "article_metadata_only" if row["xml_articles"] else "article_xml_missing")
        totals[row["status"]] += 1
        if row["initially_deferred"]:
            deferred_totals[row["status"]] += 1
    return {"run_id": run, "checked_xml_assets": checked, "xml_parse_errors": parse_errors, "totals": dict(totals),
            "nonarticle_metadata": nonarticle_metadata,
            "initially_deferred_totals": dict(deferred_totals), "input_sha256": fingerprints,
            "articles": list(records.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT.parent):
        parser.error("Audit output must stay in this project's recovery directory")
    results = []
    for run in RUNS:
        result = audit_run(run)
        results.append(result)
        print(json.dumps({k: result[k] for k in ("run_id", "checked_xml_assets", "totals", "initially_deferred_totals")}), flush=True)
    total, deferred = Counter(), Counter()
    for result in results:
        total.update(result["totals"])
        deferred.update(result["initially_deferred_totals"])
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checked_at": datetime.now(timezone.utc).isoformat(), "totals": dict(total),
        "initially_deferred_totals": dict(deferred), "runs": results,
        "method": "verified_raw_xml_article_identity_body_and_source_linked_parsed_documents",
        "originals_modified": False, "database_modified": False}
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"totals": dict(total), "initially_deferred_totals": dict(deferred), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
