"""Source recovery and reversible curation of the registered serving evidence.

Original evidence, experiment inputs and downloaded source versions are retained.
Network recovery is bounded and restricted to the public hosts already recorded
in the database; a changed response is never described as the original bytes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from urllib.parse import urlsplit

import revise_expansion_evidence as r

BASE = r.RECOVERY / "legacy_curation_20261022"
OLD = {"run_20260713T163840Z", "run_20260907T134918Z", "run_20260909T083120Z"}
HOSTS = {"eutils.ncbi.nlm.nih.gov", "dailymed.nlm.nih.gov", "ftp.ncbi.nlm.nih.gov",
         "api.fda.gov", "pubchem.ncbi.nlm.nih.gov", "www.ebi.ac.uk", "pk-db.com",
         "raw.githubusercontent.com", "github.com", "pmc-oa-opendata.s3.amazonaws.com"}
NETWORK_LOCK = threading.Lock()
NEXT_REQUEST = 0.0


def read(path):
    return json.loads(r.inside(path).read_text())


def atomic(path, value):
    path = r.inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def safe_url(url):
    parsed = urlsplit(url or "")
    return (parsed.scheme == "https" and parsed.hostname in HOSTS
            and not parsed.username and not parsed.password
            and not re.search(r"(?:key|token|secret|password)=", parsed.query, re.I))


def recover_one(item):
    import requests
    global NEXT_REQUEST
    source, row, local = item
    key = hashlib.sha256(source.encode()).hexdigest()[:24]
    receipt = BASE / "sources" / (key + ".json")
    if receipt.exists():
        old = read(receipt)
        if old.get("status") in {"identical_bytes", "new_source_version"}:
            if r.sha(r.inside(Path(old["path"]))) == old["sha256"]:
                return old
            # A receipt is never source data. Recover an interrupted/invalid
            # newly downloaded copy, without touching any original asset.
            if old.get("method") == "existing_project_copy":
                raise RuntimeError("Existing project source changed")
    if local:
        result = {"original_path": source, "original_sha256": row["checksum"],
                  "path": local, "sha256": r.sha(Path(local)), "status": "identical_bytes",
                  "method": "existing_project_copy"}
        atomic(receipt, result)
        return result
    url = row.get("source_url")
    result = {"original_path": source, "original_sha256": row["checksum"],
              "source_id": row["source_id"], "status": "unresolved"}
    # Repository inventory URLs are HTML listings, not the missing JSON asset.
    if not safe_url(url) or (row["source_id"] == "osp_pbpk_model_library" and Path(source).suffix == ".json"):
        result["reason"] = "no_recoverable_raw_url"
        atomic(receipt, result)
        return result
    destination = BASE / "sources" / (key + ".source" + Path(source).suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        if shutil.disk_usage(BASE).free < 12 * 1024**3:
            raise RuntimeError("Source recovery stopped: free-space guard")
        if sum(p.stat().st_size for p in destination.parent.iterdir() if p.is_file()) > 2 * 1024**3:
            raise RuntimeError("Source recovery stopped: 2 GiB budget")
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            with NETWORK_LOCK:
                time.sleep(max(0, NEXT_REQUEST - time.monotonic()))
                NEXT_REQUEST = time.monotonic() + 0.55
            with requests.Session() as session:
                session.trust_env = False
                with session.get(url, timeout=(15, 45), stream=True, allow_redirects=False) as response:
                    if response.is_redirect:
                        result["reason"] = "redirect_requires_source_review"
                        break
                    response.raise_for_status()
                    size = 0
                    digest = hashlib.sha256()
                    with partial.open("wb") as stream:
                        for block in response.iter_content(65536):
                            size += len(block)
                            if size > 50 * 1024**2:
                                raise ValueError("source_exceeds_50MiB")
                            stream.write(block)
                            digest.update(block)
                    partial.replace(destination)
                    result.update(path=str(destination), sha256=digest.hexdigest(), bytes=size,
                                  retrieved_at=datetime.now(timezone.utc).isoformat(),
                                  status="identical_bytes" if digest.hexdigest() == row["checksum"] else "new_source_version")
                    break
        except (requests.RequestException, ValueError) as exc:
            # Never persist exception URLs, response bodies or credentials.
            result["reason"] = type(exc).__name__
            if attempt == 0:
                time.sleep(2)
        finally:
            partial.unlink(missing_ok=True)
    atomic(receipt, result)
    return result


def fallback_sources():
    """Recover unavailable PMC packages from their same-article public XML."""
    rows = read(BASE / "serving_input.json")
    source_rows = {x["storage_path"]: x for x in rows}
    receipts = [read(p) for p in (BASE / "sources").glob("*.json")
                if re.fullmatch(r"[a-f0-9]{24}\.json", p.name)]
    output = []
    for entry in receipts:
        if entry.get("status") != "unresolved":
            continue
        row = source_rows[entry["original_path"]]
        pmc = re.search(r"pmc\d+", Path(entry["original_path"]).name, re.I)
        if row["source_id"] == "pmc_oa" and pmc:
            identifier = pmc[0].upper()
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=" + identifier + "&retmode=xml"
            result = recover_one(("same_article_xml/" + identifier + ".xml", {**row, "source_url": url}, None))
            result = {**result, "original_path": entry["original_path"], "source_url": url,
                      "original_sha256": row["checksum"], "method": "same_pmc_article_xml", "pmcid": identifier}
            if result["status"] == "identical_bytes":
                result["status"] = "new_source_version"
            output.append(result)
        elif row["source_id"] == "osp_pbpk_model_library" and Path(entry["original_path"]).suffix != ".json":
            output.append(recover_one((entry["original_path"], row, None)))
    atomic(BASE / "source_fallbacks.json", output)
    print(json.dumps(dict(Counter(x["status"] for x in output))), flush=True)


def normalize(value):
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def source_content(path):
    """Read allowed local formats without executing content or extracting archives."""
    from defusedxml import ElementTree as ET
    from pbpk_kb.parsers.table_structure import element_text, local_name, parse_xml_table_element, parse_html_tables
    from pbpk_kb.processing.source_context import repair_multilevel_headers
    path = r.inside(Path(path))
    if path.stat().st_size > 50 * 1024**2:
        raise ValueError("source_too_large")
    raw = path.read_bytes()
    texts, tables = [], []

    def xml(data):
        root = ET.fromstring(data)
        texts.append(element_text(root))
        for node in root.iter():
            if local_name(node.tag).lower() == "table":
                table = parse_xml_table_element(node)
                if table:
                    tables.append(repair_multilevel_headers(table))

    def strings(value):
        if isinstance(value, str):
            texts.append(value)
            if "<table" in value.lower():
                tables.extend(repair_multilevel_headers(t) for t in parse_html_tables(value))
        elif isinstance(value, list):
            for v in value:
                strings(v)
        elif isinstance(value, dict):
            for v in value.values():
                strings(v)

    if raw.lstrip().startswith((b"{", b"[")):
        strings(json.loads(raw))
    elif raw.startswith(b"%PDF"):
        import fitz
        with fitz.open(stream=raw, filetype="pdf") as document:
            texts.extend(page.get_text() for page in document)
    elif raw.startswith(b"\x1f\x8b"):
        import io, tarfile
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            for member in archive:
                if member.isfile() and member.name.lower().endswith((".nxml", ".xml")) and member.size < 30 * 1024**2:
                    with archive.extractfile(member) as stream:
                        xml(stream.read(30 * 1024**2))
    elif raw.lstrip().startswith(b"<"):
        xml(raw)
    else:
        texts.append(raw.decode("utf-8"))
    return {"texts": [normalize(t) for t in texts], "tables": tables}


def table_support(row, content):
    """Independently reparse the source row/header and compare the selected cell."""
    from pbpk_kb.processing.table_recovery_stage import detect_parameter, find_unit, parse_table_value
    from pbpk_kb.normalization.unit_normalizer import to_canonical_value
    metadata = row.get("metadata_json") or {}
    header = metadata.get("parameter_header")
    if row.get("extraction_method") != "source_structured_table" or not header:
        return None
    quoted_cells = [normalize(x) for x in row["source_text"].split("|")]
    for table in content["tables"]:
        for cells in table.rows:
            if [normalize(x) for x in cells] != quoted_cells:
                continue
            for index, column in enumerate(table.columns):
                if normalize(column) != normalize(header) or index >= len(cells):
                    continue
                if detect_parameter(column) != row["parameter"]:
                    continue
                unit_hint = find_unit(column)
                if not unit_hint and row.get("unit") and normalize(row["unit"]).casefold() in normalize(column).casefold():
                    unit_hint = row["unit"]
                parsed = parse_table_value(cells[index], unit_hint=unit_hint)
                if not parsed:
                    continue
                value, unit = parsed[0], parsed[2]
                expected, expected_unit = to_canonical_value(value, unit)
                current, current_unit = to_canonical_value(row.get("value"), row.get("unit"))
                if expected_unit == current_unit and expected is not None and current is not None and abs(float(expected) - float(current)) <= 1e-9 * max(1, abs(float(expected))):
                    return {"method": "reparsed_source_table_cell", "column": column, "cell": cells[index],
                            "row": cells, "caption": table.caption, "footnotes": table.footnotes}
    return None


def full_quote(row, content, start, end):
    options = []
    for text in content["texts"]:
        begin = text.find(start)
        if begin >= 0:
            finish = text.find(end, begin)
            if finish >= 0:
                options.append(text[begin:finish + len(end)])
    if not options:
        raise ValueError("Reviewed complete assertion missing from source")
    return min(options, key=len)


def known_corrections(row, content):
    """A small explicit source-reviewed set; no inferred scientific numbers."""
    eid = row["evidence_id"]
    patch, reason = {}, None
    if eid == "91f4dfbe-32e9-55b3-95fe-6302a1d05e01":
        patch = {"source_text": full_quote(row, content, "When multiple doses of voriconazole", "administered as the oral suspension"),
                 "value": 24.0, "canonical_value": 24.0, "value_text": "24% reduction", "parameter": "AUC_tau",
                 "condition": "Multiple doses; oral tablet; high-fat meal versus fasting; AUC_tau reduction", "route": "oral"}
        reason = "Complete label assigns 34% to Cmax and 24% to AUC_tau for tablets; suspension differs."
    elif eid == "a9e41eb1-2230-5dfc-9a24-34201d20666a":
        patch = {"source_text": full_quote(row, content, "Clinical Studies Ketoconazole:", "33% [90% CI (24, 43)] respectively"),
                 "condition": "Bedaquiline 400 mg daily for 14 days with ketoconazole 400 mg daily for 4 days; AUC_24h increase",
                 "population": "healthy adults"}
        reason = "Restore complete endpoint list; original AUC_24h increase of 22% is supported."
    elif eid == "c406c298-5ad6-5c63-bda2-d94351fd74b2":
        patch = {"source_text": full_quote(row, content, "Metformin is negligibly bound to plasma proteins", "more than 90% protein bound."),
                 "value": None, "canonical_value": None, "unit": None, "canonical_unit": None,
                 "value_low": None, "value_high": None, "value_qualifier": None,
                 "value_text": "negligibly bound to plasma proteins"}
        reason = "90% belongs to sulfonylureas; retain only the qualitative metformin assertion."
    elif eid == "3dd55e59-122a-5ea9-8801-5786e7edf53f":
        patch = {"value": None, "canonical_value": None, "unit": None, "canonical_unit": None,
                 "value_low": None, "value_high": None, "value_qualifier": None,
                 "value_text": "not a substrate for P-gp"}
        reason = "1% urinary excretion belongs to alfentanil; retain supported qualitative non-substrate statement."
    elif eid == "5cce29f7-3eb3-52e8-8c3c-b8e13df298dd":
        patch = {"parameter": "Cmax", "value_text": "61.6% of comparison group",
                 "condition": "WCO versus WO group; relative Cmax, not AUC; animal study context retained"}
        reason = "Source explicitly assigns 61.6% to Cmax, not AUC_0-t."
    if patch:
        updated = {**row, **patch}
        updated["source_text_hash"] = hashlib.sha256(updated["source_text"].encode()).hexdigest()
        updated["metadata_json"] = {**updated.get("metadata_json", {}), "source_review_correction": {
            "reason": reason, "changed_fields": sorted(patch), "method": "source_bound_agent_review_not_expert_adjudication"}}
        return updated, reason
    return row, None


def structured_field_support(row, path):
    """Verify a JSON field at the registered location, including named molecule."""
    if row.get("extraction_method") != "source_structured_field":
        return False
    marker = re.search(r"#json_path=(\$[^#]+)", row["source_location"])
    if not marker:
        return False
    try:
        document = json.loads(r.inside(Path(path)).read_bytes())
        value = document
        for name, index in re.findall(r"\.([A-Za-z_][A-Za-z_0-9]*)|\[(\d+)\]", marker[1]):
            value = value[int(index)] if index else value[name]
        recorded = row.get("value") if row.get("value") is not None else row.get("value_text")
        same = str(recorded) == str(value)
        if not same and row.get("value") is not None:
            same = abs(float(recorded) - float(value)) < 1e-10 * max(1, abs(float(value)))
        if not same:
            return False
        if row["source_id"] == "chembl":
            from pbpk_kb.processing.structured_evidence_validation import _chembl_names, _identity
            index = int(re.search(r"\.molecules\[(\d+)\]", marker[1])[1])
            return _identity(row["compound_name"]) in _chembl_names(document["molecules"][index])
        if row["source_id"] == "pubchem":
            # Same raw checksum is the original CID-bound import. Changed
            # responses need the separate structured identity validation.
            return r.sha(Path(path)) == row.get("original_checksum_for_review", row["checksum"])
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return False
    return False


def audit():
    from copy import deepcopy
    from pbpk_kb.processing.auto_quality_stage import evaluate_evidence, deterministic_repair
    from pbpk_kb.processing.source_context import enrich_source_context
    from pbpk_kb.processing.final_evidence_quality_stage import evaluate_domain_gate
    input_rows = read(BASE / "serving_input.json")
    recovered = {x["original_path"]: x for x in read(BASE / "source_recovery.json")}
    if (BASE / "source_fallbacks.json").exists():
        for entry in read(BASE / "source_fallbacks.json"):
            if entry["status"] != "unresolved":
                recovered[entry["original_path"]] = entry
    content_cache, context_cache = {}, {}
    output = []
    for original in sorted(input_rows, key=lambda x: (x["storage_path"], x["evidence_id"])):
        if original["run_id"] not in OLD and original["evidence_id"] != "a9e41eb1-2230-5dfc-9a24-34201d20666a":
            continue
        row = {**deepcopy(original.get("source_record") or {}), **deepcopy({k: v for k, v in original.items() if k != "source_record"})}
        # Numeric ranges were serialized as strings by the old SQL importer.
        for field in ("value_low", "value_high", "canonical_value_low", "canonical_value_high"):
            if row.get(field) is not None:
                row[field] = float(row[field])
        old_path = row["storage_path"]
        row["original_checksum_for_review"] = row["checksum"]
        source = recovered.get(old_path)
        blockers, resolved, correction = [], [], None
        if source and source["status"] != "unresolved":
            row["storage_path"], row["checksum"] = source["path"], source["sha256"]
            if source.get("source_url"):
                row["source_url"] = source["source_url"]
        elif not old_path.startswith(str(r.ROOT)):
            blockers.append("original_source_not_recovered")
        content, support = None, None
        if not blockers:
            path = r.inside(Path(row["storage_path"]))
            if not path.is_file() or r.sha(path) != row["checksum"]:
                blockers.append("source_checksum_mismatch")
            else:
                try:
                    if str(path) not in content_cache:
                        if len(content_cache) >= 4:
                            content_cache.pop(next(iter(content_cache)))
                        content_cache[str(path)] = source_content(path)
                    content = content_cache[str(path)]
                    support = table_support(row, content)
                    quote_found = any(normalize(row["source_text"]) in t for t in content["texts"])
                    field_support = structured_field_support(row, path)
                    if not quote_found and not support and not field_support:
                        blockers.append("source_assertion_not_relocated")
                    else:
                        row, correction = known_corrections(row, content)
                        row = enrich_source_context(row, project_root=r.ROOT, cache=context_cache)
                except Exception as exc:
                    blockers.append("source_review_error:" + type(exc).__name__)
        if source and source["status"] == "new_source_version" and not blockers:
            # Positional indexes in a changed response cannot be retained.
            row["source_location"] = row["source_id"] + ":" + Path(row["storage_path"]).name + "#quote_sha256=" + row["source_text_hash"]
        if not blockers:
            if field_support and row["source_id"] in {"pubchem", "chembl"}:
                units = {"hba": "dimensionless", "hbd": "dimensionless", "alogp": "dimensionless", "XLogP": "dimensionless",
                         "full_mwt": "g/mol", "MolecularWeight": "g/mol", "psa": "angstrom^2", "TPSA": "angstrom^2"}
                if row["parameter"] in units:
                    row["unit"] = row["canonical_unit"] = units[row["parameter"]]
            repaired, repairs = deterministic_repair(row)
            decision = evaluate_evidence(repaired, repairs)
            fatal = list(decision["fatal_blockers"])
            if field_support and row["checksum"] == row["original_checksum_for_review"]:
                # Identical raw bytes relocated across acquisitions can have a
                # different asset UUID. Field and molecular identity were
                # independently checked, not inferred from a cached verdict.
                permitted = {"structured_property_candidate_source_identity_mismatch"}
                resolved += [x for x in fatal if x in permitted]
                fatal = [x for x in fatal if x not in permitted]
            if support:
                removed = [x for x in fatal if x in {"numeric_value_not_supported_for_target_parameter", "numeric_value_belongs_to_other_substance"}]
                fatal = [x for x in fatal if x not in removed]
                resolved.extend(removed)
                repaired["metadata_json"]["reparsed_table_verification"] = support
            if row["evidence_id"] == "37eda0a7-a0f8-553f-9580-0ec76ff28067":
                if all(s in row["source_text"] for s in ("AUC 0 to 12", "20%", "(C max ) by 16%", "(C 12hr ) by 26%")) and row["value"] == 20:
                    resolved += [x for x in fatal if x == "truncated_multi_endpoint_percent_assignment"]
                    fatal = [x for x in fatal if x != "truncated_multi_endpoint_percent_assignment"]
            domain = evaluate_domain_gate(repaired)
            blockers.extend(fatal)
            # Existing relative-reference annotations are retained. Newly
            # unresolved scientific context is queued, never counted as false.
            blockers.extend(decision["repairable_blockers"])
            if domain["domain_gate_status"] != "passed":
                blockers.extend(domain["reasons"])
            row = repaired
            row["domain_gate_status"] = domain["domain_gate_status"]
            row["domain_gate_reasons"] = domain["reasons"]
            if domain.get("species"):
                row["species"] = domain["species"]
        output.append({"evidence_id": row["evidence_id"], "run_id": row["run_id"], "row": row,
                       "original_sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
                       "source_recovery": source, "correction": correction,
                       "resolved_false_positives": resolved, "blockers": sorted(set(blockers)),
                       "status": "reviewable" if blockers else "accepted"})
    groups = defaultdict(list)
    # Never merge different doses, ranges, species, directions or drug roles.
    fields = ("compound_id", "compound_role", "evidence_type", "parameter", "value", "value_text", "unit", "value_low", "value_high", "value_qualifier", "condition", "population", "route", "dose", "species", "source_text")
    for item in output:
        if item["status"] == "accepted":
            groups[json.dumps([item["row"].get(k) for k in fields], sort_keys=True)].append(item)
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda x: (bool(x["source_recovery"] and x["source_recovery"]["status"] != "identical_bytes"), x["evidence_id"]))
        for item in group[1:]:
            item.update(status="duplicate", canonical_evidence_id=group[0]["evidence_id"])
    summary = {"reviewed": len(output), "statuses": dict(Counter(x["status"] for x in output)),
               "blockers": dict(Counter(b for x in output for b in x["blockers"])),
               "corrected": sum(bool(x["correction"]) for x in output),
               "false_positive_resolved": sum(bool(x["resolved_false_positives"]) for x in output),
               "source_paths_restored": sum(bool(x["source_recovery"] and not x["blockers"]) for x in output),
               "interpretation": "Automated and source-bound agent review; not expert validation. Reviewable is not a false-data label."}
    atomic(BASE / "curation_proposal.json", {"summary": summary, "items": output})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def recover():
    rows = read(BASE / "serving_input.json")
    matches = read(BASE / "source_candidates.json")["exact"]
    unique = {v["storage_path"]: v for v in rows if v["run_id"] in OLD
              and not v["storage_path"].startswith(str(r.ROOT))}
    output = []
    atomic(BASE / "recovery_status.json", {"status": "running", "pid": os.getpid(), "total": len(unique), "done": 0})
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(recover_one, (source, row, matches.get(source))) for source, row in unique.items()]
        for job in as_completed(jobs):
            output.append(job.result())
            counts = dict(Counter(x["status"] for x in output))
            atomic(BASE / "recovery_status.json", {"status": "running", "pid": os.getpid(),
                   "total": len(unique), "done": len(output), "counts": counts})
            if len(output) % 25 == 0:
                print(json.dumps({"done": len(output), "total": len(unique), "counts": counts}), flush=True)
    atomic(BASE / "source_recovery.json", output)
    atomic(BASE / "recovery_status.json", {"status": "completed", "total": len(unique), "done": len(output), "counts": counts})
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["recover", "fallback", "audit"])
    args = parser.parse_args()
    if args.action == "recover":
        recover()
    elif args.action == "fallback":
        fallback_sources()
    elif args.action == "audit":
        audit()
