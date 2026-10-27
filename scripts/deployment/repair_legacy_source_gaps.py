"""Resolve old source/format gaps without weakening the serving quality gates.

Rebind API facts to newly downloaded, registered, identity-checked sources;
relocate PDF quotations only through exact typographic normalization. Original
sources and the first curation receipt remain immutable. No fuzzy text match.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import unicodedata
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5

import apply_legacy_curation as apply_tool
import curate_legacy_evidence as curate
import revise_expansion_evidence as r
from verify_expansion_batch import _registered_database, _local_query_vectors
from pbpk_kb.processing.auto_quality_stage import deterministic_repair, evaluate_evidence
from pbpk_kb.processing.final_evidence_quality_stage import evaluate_domain_gate
from pbpk_kb.processing.structured_evidence_validation import validate_structured_evidence

OUT = curate.BASE / "source_gap_followup"
REVISION = "legacy_source_gap_followup_20261022"
TARGET_FLAGS = {"source_assertion_not_relocated", "structured_property_source_missing_or_oversized", "explicit_animal_species_in_clinical_pk"}


def endpoint_semantics_blockers(row):
    """A source-located number can still have the wrong scientific meaning."""
    text = str(row.get("source_text") or "")
    parameter = str(row.get("parameter") or "")
    value = row.get("value")
    blockers = []
    if value is not None and str(row.get("unit") or "").strip() == "%":
        number = re.escape(f"{float(value):g}")
        if not re.search(r"variation|variability|\bcv\b", parameter, re.I) and re.search(r"(?:coefficient\s+of\s+variation\s*\(CV\)|\bCV)\s*[<=>≤≥]*\s*" + number + r"\s*%", text, re.I):
            blockers.append("variability_statistic_not_parameter_value")
        if re.search(r"fraction[ _]unbound", parameter, re.I) and re.search(number + r"\s*%\s+increase\s+in\s+fraction\s+unbound", text, re.I):
            blockers.append("relative_model_assumption_not_absolute_parameter")
    # Source review identified these clauses as insufficient for their current
    # target role/endpoint. IDs are retained with a precise review reason.
    checked = {
        "322363c7-0810-5b74-8ee1-83e24cadf0e3": "caffeine_as_interacting_agent_endpoint_owner_unclear",
        "54a0ff2c-8df7-5104-bf96-8c5b74ea2c29": "caffeine_as_interacting_agent_endpoint_owner_unclear",
        "91491677-81dd-5435-a1c7-c2d37528f5cd": "reference_title_without_result_context",
        "ca7c331d-2334-56bc-aa6a-7854cdd916fd": "relative_bioavailability_quote_missing_direction_and_counterparty",
        "d120c9d0-952f-5e6e-96a0-c447e29be272": "population_pk_vs_pbpk_model_identity_requires_review",
    }
    if row.get("evidence_id") in checked: blockers.append(checked[row["evidence_id"]])
    return blockers


def typographic_key(text):
    """Keep all letters, numbers, punctuation, signs and units; map offsets."""
    chars, offsets = [], []
    # PDF line breaks can split a word. This is the sole character deletion
    # besides whitespace/soft hyphens; do not remove citations or numbers.
    joins = {m.start() for m in re.finditer(r"(?<=[a-z])[-\u00ad](?=\s+[a-z])", text)}
    for pos, char in enumerate(text):
        if pos in joins or char == "\u00ad":
            continue
        for normalized in unicodedata.normalize("NFKC", char):
            if normalized.isspace():
                continue
            chars.append({"−": "-", "–": "-", "‐": "-", "‑": "-", "‘": "'", "’": "'", "“": '"', "”": '"'}.get(normalized, normalized))
            offsets.append(pos)
    return "".join(chars), offsets


def locate_quote(quote_text, texts):
    target, _ = typographic_key(quote_text)
    if len(target) < 30:
        return None
    matches = set()
    for text in texts:
        if quote_text in text:
            if text.count(quote_text) != 1: return None
            matches.add(quote_text)
            continue
        # Cheap rejection before constructing a character-offset map for a
        # potentially long article. A successful match is still checked by
        # the exact mapped transformation below.
        fast = unicodedata.normalize("NFKC", text)
        fast = re.sub(r"(?<=[a-z])[-\u00ad](?=\s+[a-z])", "", fast)
        fast = re.sub(r"\s+|\u00ad", "", fast).translate(str.maketrans({"−": "-", "–": "-", "‐": "-", "‑": "-", "‘": "'", "’": "'", "“": '"', "”": '"'}))
        if target not in fast: continue
        key, offsets = typographic_key(text)
        start = key.find(target)
        if start >= 0:
            if key.find(target, start+1) >= 0:
                return None
            matches.add(text[offsets[start]:offsets[start+len(target)-1]+1])
    return next(iter(matches)) if len(matches) == 1 else None


def fetch_api(compound, source, operation):
    import requests
    directory = OUT / "api_sources" / compound / source
    name = ("properties_by_name" if operation == "property" else "synonyms_by_name") if source == "pubchem" else "molecule_search"
    path = directory / (name + "_revalidated.json")
    sidecar_path = Path(str(path) + ".metadata.json")
    if path.exists() and sidecar_path.exists():
        sidecar = curate.read(sidecar_path)
        if r.sha(path) != sidecar["checksum"]: raise RuntimeError("API source changed")
        return path, sidecar
    if shutil.disk_usage(OUT).free < 12 * 1024**3: raise RuntimeError("Source download free-space guard")
    if source == "pubchem":
        suffix = "property/MolecularWeight,TPSA,XLogP,InChIKey/JSON" if operation == "property" else "synonyms/JSON"
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/" + quote(compound, safe="") + "/" + suffix
    else:
        url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q=" + quote(compound, safe="")
    if not curate.safe_url(url): raise RuntimeError("Unregistered public source")
    directory.mkdir(parents=True, exist_ok=True)
    time.sleep(.6)
    with requests.Session() as session:
        session.trust_env = False
        with session.get(url, timeout=(15, 45), stream=True, allow_redirects=False) as response:
            if response.status_code != 200: raise RuntimeError("API source response status " + str(response.status_code))
            blocks, size = [], 0
            for block in response.iter_content(65536):
                size += len(block)
                if size > 16*1024**2: raise RuntimeError("API source size limit")
                blocks.append(block)
            data = b"".join(blocks)
    json.loads(data)
    digest = hashlib.sha256(data).hexdigest()
    path.write_bytes(data)
    sidecar = {"source_id": source, "compound": compound, "status_code": 200, "url": url,
               "checksum": digest, "byte_count": len(data), "storage_path": str(path),
               "asset_id": str(uuid5(NAMESPACE_URL, REVISION + "|" + compound + "|" + source + "|" + digest)),
               "retrieved_at": datetime.now(timezone.utc).isoformat()}
    curate.atomic(sidecar_path, sidecar)
    return path, sidecar


def structured_rebind(row):
    source, compound = row["source_id"], row["compound_name"]
    # PubChem structure and synonyms are also needed to resolve ChEMBL's
    # shared salt/stereoisomer aliases; the existing validator enforces this.
    fetch_api(compound, "pubchem", "synonyms")
    fetch_api(compound, "pubchem", "property")
    path, sidecar = fetch_api(compound, source, "property")
    payload = curate.read(path)
    field = row["parameter"]
    metadata = dict(row["metadata_json"])
    if source == "pubchem":
        props = payload["PropertyTable"]["Properties"]
        if len(props) != 1: raise ValueError("Ambiguous current PubChem response")
        item, pointer = props[0], "$.PropertyTable.Properties[0]." + field
        metadata["pubchem_cid"] = item["CID"]
        raw_value = item[field]
    else:
        expected = metadata.get("molecule_chembl_id")
        matches = [(i, x) for i, x in enumerate(payload["molecules"]) if x.get("molecule_chembl_id") == expected]
        if len(matches) != 1: raise ValueError("Original ChEMBL molecule is no longer uniquely returned")
        index, item = matches[0]
        raw_value = item["molecule_properties"][field]
        pointer = f"$.molecules[{index}].molecule_properties.{field}"
    if abs(float(raw_value)-float(row["value"])) > 1e-12*max(1, abs(float(raw_value))):
        raise ValueError("Provider value changed; not an old-source relocation")
    metadata["source_gap_repair"] = {"method": "new_provider_response_same_property_and_identity", "original_checksum": row["checksum"]}
    row.update(storage_path=str(path), checksum=sidecar["checksum"], source_url=sidecar["url"],
               raw_asset_id=sidecar["asset_id"], source_location=f"{source}:{path.name}#json_path={pointer}",
               source_text=str(raw_value), source_text_hash=hashlib.sha256(str(raw_value).encode()).hexdigest(), metadata_json=metadata,
               parsed_document_id=None, parsed_table_id=None, chunk_id=None)
    proof = validate_structured_evidence(row)
    if not proof or not proof["accepted"]: raise ValueError((proof or {}).get("error", "No supported structured validator"))
    return row


def review():
    import psycopg
    OUT.mkdir(parents=True, exist_ok=True)
    proposal = curate.read(curate.BASE / "curation_proposal.json")
    selected = {x["evidence_id"]: x for x in proposal["items"] if x["status"] == "reviewable" and set(x["blockers"]) & TARGET_FLAGS}
    originals_path = OUT / "input.json"
    if originals_path.exists():
        originals = curate.read(originals_path)
    else:
        with psycopg.connect(r.imp.psycopg_dsn(r.registered_url()), options="-c default_transaction_read_only=on") as c:
            originals = [x[0] for x in c.execute("SELECT to_jsonb(t) FROM reviewable_evidence t WHERE evidence_id::text=ANY(%s)", (sorted(selected),))]
            c.rollback()
        r.write_new(originals_path, originals)
    results, content_cache, context_cache = [], {}, {}
    for original in originals:
        row = {**deepcopy(original.get("source_record") or {}), **deepcopy({k:v for k,v in original.items() if k != "source_record"})}
        eid = row["evidence_id"]
        blockers = selected[eid]["blockers"]
        repairs = []
        try:
            for field in ("value_low", "value_high", "canonical_value_low", "canonical_value_high"):
                if row.get(field) is not None: row[field] = float(row[field])
            if "structured_property_source_missing_or_oversized" in blockers:
                row = structured_rebind(row); repairs.append("registered_provider_identity_and_value")
            else:
                path = r.inside(Path(row["storage_path"]))
                if r.sha(path) != row["checksum"]: raise RuntimeError("Source checksum changed")
                if str(path) not in content_cache:
                    if len(content_cache) >= 4: content_cache.pop(next(iter(content_cache)))
                    content_cache[str(path)] = curate.source_content(path)
                content = content_cache[str(path)]
                match = locate_quote(row["source_text"], content["texts"])
                if match is None: raise ValueError("No unique exact typographic source match")
                if row["source_text"] != match:
                    row["source_text"] = match
                    row["source_text_hash"] = hashlib.sha256(match.encode()).hexdigest()
                    row["source_location"] = row["source_id"] + ":" + path.name + "#quote_sha256=" + row["source_text_hash"]
                    repairs.append("pdf_xml_typography_only_quote_relocation")
                from pbpk_kb.processing.source_context import enrich_source_context
                row = enrich_source_context(row, project_root=r.ROOT, cache=context_cache)
                # An animal paper can quote human clinical knowledge. Only
                # explicit human assertions, without animal words in the
                # statement or conditions, can override the article prior.
                if "explicit_animal_species_in_clinical_pk" in blockers:
                    from pbpk_kb.processing.source_context import species_mentions
                    human = re.search(r"\b(?:in\s+humans?|in\s+(?:healthy\s+)?(?:patients|volunteers)|healthy\s+(?:male\s+|female\s+)?(?:subjects|volunteers))\b", row["source_text"], re.I)
                    local = " ".join(str(row.get(k) or "") for k in ("source_text", "condition", "population"))
                    if human and not species_mentions(local):
                        metadata = dict(row.get("metadata_json") or {})
                        prior = dict(metadata.get("source_context") or {})
                        metadata["source_context_article_prior"] = prior
                        metadata["source_context"] = {**prior, "animal_study": False, "species": [], "species_scope": "explicit_human_assertion",
                                                      "species_scope_proof": human.group(0), "statement_sha256": row["source_text_hash"]}
                        row.update(metadata_json=metadata, species=["human"], population="human")
                        repairs.append("explicit_statement_species_over_article_species")
            row, deterministic = deterministic_repair(row)
            quality = evaluate_evidence(row, deterministic)
            domain = evaluate_domain_gate(row)
            row.update(quality)
            row["domain_gate_status"] = domain["domain_gate_status"]
            row["domain_gate_reasons"] = domain["reasons"]
            semantic_blockers = endpoint_semantics_blockers(row)
            accepted = quality["quality_status"] == "auto_validated" and domain["domain_gate_status"] == "passed" and not semantic_blockers
            results.append({"evidence_id": eid, "status": "accepted" if accepted else "reviewable", "repairs": repairs,
                            "blockers": sorted(set(quality["fatal_blockers"] + quality["repairable_blockers"] + domain["reasons"] + semantic_blockers)), "row": row})
        except Exception as exc:
            results.append({"evidence_id": eid, "status": "reviewable", "repairs": repairs, "blockers": blockers,
                            "error": str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__})
    result = {"summary": {"reviewed": len(results), "status": dict(Counter(x["status"] for x in results)),
                          "repairs": dict(Counter(k for x in results for k in x["repairs"]))}, "items": results}
    curate.atomic(OUT / "proposal.json", result)
    print(json.dumps(result["summary"]))


def apply():
    import psycopg
    from psycopg.types.json import Jsonb
    from pbpk_kb.processing.rag_index_stage import evidence_record, build_embedding_text
    if (OUT / "committed.json").exists(): raise RuntimeError("Source gap followup already committed")
    originals = {x["evidence_id"]: x for x in curate.read(OUT / "input.json")}
    proposal = curate.read(OUT / "proposal.json")
    candidates = [x for x in proposal["items"] if x["status"] == "accepted"]
    fields = ("compound_id", "compound_role", "evidence_type", "parameter", "value", "value_text", "unit", "value_low", "value_high", "value_qualifier", "condition", "population", "route", "dose", "species", "source_text")
    def key(row): return apply_tool.hash_object([row.get(k) for k in fields])
    url, config = _registered_database()
    with psycopg.connect(r.imp.psycopg_dsn(r.registered_url()), options="-c default_transaction_read_only=on") as c:
        existing = [x[0] for x in c.execute("SELECT to_jsonb(t) FROM auto_validated_evidence t")]
        registered_raw = {x["raw_document_id"]: x for (x,) in c.execute("SELECT to_jsonb(t) FROM raw_document t WHERE raw_document_id::text=ANY(%s)", (sorted({x["raw_document_id"] for x in originals.values()}),))}
        indices = dict(c.execute("SELECT run_id,max(embedding_row)+1 FROM rag_retrieval_record GROUP BY run_id"))
        c.rollback()
    known = {key({**x.get("source_record", {}), **{k:v for k,v in x.items() if k != 'source_record'}}):x["evidence_id"] for x in existing}
    items, duplicates, raw = [], [], {}
    for item in candidates:
        row = item["row"]; eid = row["evidence_id"]
        if key(row) in known:
            duplicates.append({"evidence_id": eid, "canonical_evidence_id": known[key(row)]}); continue
        known[key(row)] = eid
        if (evaluate_evidence(row, [])["quality_status"] != "auto_validated" or evaluate_domain_gate(row)["domain_gate_status"] != "passed"
                or endpoint_semantics_blockers(row)):
            raise RuntimeError("Source repair quality gate changed")
        path = r.inside(Path(row["storage_path"]))
        if r.sha(path) != row["checksum"]: raise RuntimeError("Source bytes changed after audit")
        old = originals[eid]
        registered = registered_raw.get(old["raw_document_id"], {})
        binding_changed = (registered.get("raw_storage_path") != row["storage_path"] or registered.get("raw_checksum") != row["checksum"]
                           or registered.get("source_record_id") != row["raw_asset_id"])
        if binding_changed:
            raw_id = str(uuid5(NAMESPACE_URL, REVISION + "|" + row["source_id"] + "|" + row["checksum"]))
            if row["source_id"] not in {"pubchem", "chembl"}:
                row["raw_asset_id"] = "source-followup:" + row["checksum"]
            row["raw_document_id"] = raw_id
            row["parsed_document_id"] = row["parsed_table_id"] = row["chunk_id"] = None
            raw[raw_id] = {"raw_document_id": raw_id, "source_id": row["source_id"], "source_native_id": row["raw_asset_id"],
                           "source_url": row.get("source_url"), "source_version": REVISION, "raw_storage_path": str(path),
                           "raw_checksum": row["checksum"], "raw_size_bytes": path.stat().st_size,
                           "metadata": {"original_raw_document_id": old["raw_document_id"], "original_checksum": old["checksum"], "repair": item["repairs"]}}
        items.append({**item, "resolved_false_positives": item["repairs"], "correction": {"source_repair": item["repairs"]}, "blockers": []})
    records = []
    for item in items:
        record = evidence_record(item["row"], "auto_validated")
        record["embedding_text"] = build_embedding_text(record)
        record["retrieval_record_id"] = str(uuid5(NAMESPACE_URL, REVISION + "|" + item["evidence_id"]))
        record["embedding_row"] = indices.get(record["run_id"], 0)
        indices[record["run_id"]] = record["embedding_row"] + 1
        records.append(record)
    if not items:
        curate.atomic(OUT / "committed.json", {"promoted": 0, "duplicates_not_readmitted": duplicates}); return
    vectors, revision = _local_query_vectors([x["embedding_text"] for x in records], config)
    for record in records: record["embedding"] = vectors[record["embedding_text"]]
    ids = [x["evidence_id"] for x in items]
    rag_ids = [x["retrieval_record_id"] for x in records]
    scope = {"evidence_ids": ids, "rag_ids": rag_ids, "new_rag_ids": rag_ids, "raw_ids": sorted(raw), "new_raw_ids": sorted(raw), "active_ids": ids}
    bundle = {"scope": scope, "items": items, "rag_records": records, "raw_documents": list(raw.values()),
              "proposal_sha256": r.sha(OUT / "proposal.json"), "timestamp": datetime.now(timezone.utc).isoformat()}
    r.write_new(OUT / "apply_bundle.json", bundle)
    apply_tool.REVISION = REVISION
    def mutate(c):
        apply_tool.mutate(c, bundle)
        for item in items:
            eid, row = item["evidence_id"], item["row"]
            c.execute("DELETE FROM evidence_layer_membership WHERE evidence_id::text=%s AND evidence_layer='reviewable'", (eid,))
            c.execute("INSERT INTO evidence_layer_membership(evidence_id,run_id,evidence_layer,metadata_json) VALUES (%s,%s,'auto_validated',%s)", (eid, row["run_id"], Jsonb({"source_gap_repair": item["repairs"]})))
    with psycopg.connect(r.imp.psycopg_dsn(r.registered_url()), autocommit=True, options="-c statement_timeout=180000 -c lock_timeout=10000") as c:
        class Rollback(Exception): pass
        try:
            with c.transaction():
                r.lock_project_tables(c)
                current = {x["evidence_id"]:x for (x,) in c.execute("SELECT to_jsonb(t) FROM reviewable_evidence t WHERE evidence_id::text=ANY(%s)", (ids,))}
                if any(apply_tool.hash_object(current.get(eid)) != apply_tool.hash_object(originals[eid]) for eid in ids): raise RuntimeError("Source followup snapshot is stale")
                before, protected = apply_tool.snapshot(c, scope), apply_tool.snapshot(c, scope, True)
                r.write_new(OUT / "before.json", {"scope": scope, "rows": before, "protected": protected})
                mutate(c); apply_tool.verify(c, bundle)
                r.assert_protected(protected, apply_tool.snapshot(c, scope, True))
                raise Rollback()
        except Rollback:
            if apply_tool.snapshot(c, scope) != before: raise RuntimeError("Source followup rollback failed")
        with c.transaction():
            r.lock_project_tables(c)
            if apply_tool.snapshot(c, scope) != before: raise RuntimeError("Source followup changed after rehearsal")
            # The seven-batch controller may commit outside our scope between
            # transactions. Protect the current outside state, not an old one.
            protected = apply_tool.snapshot(c, scope, True)
            mutate(c); counts = apply_tool.verify(c, bundle)
            r.assert_protected(protected, apply_tool.snapshot(c, scope, True))
            r.write_new(OUT / "after.json", {"rows": apply_tool.snapshot(c, scope)})
    receipt = {"promoted": len(items), "counts": counts, "duplicates_not_readmitted": duplicates,
               "raw_sources": len(raw), "embedding_revision": revision, "rollback_rehearsed": True,
               "protected_rows_verified": sum(map(len, protected.values())), "source_rules_rechecked": True}
    r.write_new(OUT / "committed.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("action", choices=["review", "apply"])
    args = p.parse_args()
    review() if args.action == "review" else apply()
