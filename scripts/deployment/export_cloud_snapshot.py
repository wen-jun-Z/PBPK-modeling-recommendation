"""Export a compact, read-only paper UI snapshot from this project's data.

Run using runtime/app-env/bin/python. No service, database or frozen research
asset is changed. The database reads share one repeatable-read transaction;
portable files are replaced only after all native source pins are verified.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit, parse_qsl, parse_qs

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "cloud/data"

# Display metadata from legacy saved jobs; original scientific excerpts are unchanged.
ENGLISH_METADATA = {
    "\u8fd9\u662f\u5f53\u524d PBPK-Match light \u7684\u7814\u7a76\u63a8\u7406\uff1b\u4fdd\u5b58\u7684\u5206\u7ec4\u6743\u91cd\u6ca1\u6709\u88ab\u5ba3\u79f0\u4e3a\u72ec\u7acb\u9a8c\u8bc1\u7684\u751f\u4ea7\u6a21\u578b\u3002": "Research inference with the current PBPK-Match light model; the saved grouped weights are not claimed to be independently validated production models.",
    "\u5206\u6570\u7528\u4e8e\u5019\u9009\u6392\u5e8f\uff1bY/N/U \u8868\u793a\u6587\u6863\u652f\u6301\u6982\u7387\uff0c\u4e0d\u4ee3\u8868\u771f\u5b9e\u9884\u6d4b\u51c6\u786e\u7387\u3002": "Scores rank candidates. Y/N/U probabilities describe documentary support, not actual predictive accuracy.",
    "\u7528\u9014\u68c0\u67e5\u5355\u72ec\u4fdd\u7559\uff0c\u7814\u7a76\u6392\u540d\u4e0d\u8986\u76d6\u8bb8\u53ef\u3001\u9002\u7528\u8303\u56f4\u6216\u6267\u884c\u9a8c\u8bc1\u6761\u4ef6\u3002": "Applicability checks remain separate; research rankings do not override licensing, scope, or execution-validation requirements.",
    "\u672c\u6b21\u6ca1\u6709\u8fdb\u884c PBPK \u4eff\u771f\u3002\u6d53\u5ea6\u66f2\u7ebf\u9875\u9762\u5c55\u793a\u5df2\u5b8c\u6210\u7814\u7a76\u4e2d\u7684\u56fa\u5b9a\u6848\u4f8b\u3002": "This recommendation did not run PBPK simulation. Concentration curves come from fixed cases in completed studies.",
    "\u5148\u6309\u7528\u9014\u68c0\u67e5\u5206\u5c42\uff0c\u518d\u5728\u540c\u5c42\u5185\u6309\u7814\u7a76\u5206\u6570\u6392\u5e8f\uff1b\u5f85\u9a8c\u8bc1\u4e0d\u7b49\u4e8e\u53ef\u76f4\u63a5\u4f7f\u7528\u3002\u539f\u59cb\u7814\u7a76\u6392\u540d\u5b8c\u6574\u4fdd\u7559\u5728\u4e0b\u8f7d\u4e2d\u3002": "Candidates are grouped by applicability, then ranked within each tier by research score. Pending validation does not imply direct reuse. Downloads retain the complete raw ranking.",
    "\u540c\u836f\u6a21\u578b\uff0c\u9700\u8865\u5145\u9a8c\u8bc1": "Same-drug model; further qualification required",
    "\u5f53\u524d\u4e0d\u53ef\u4f7f\u7528": "Not applicable to the current request",
    "\u5df2\u4fdd\u5b58\u7684\u65f6\u95f4\u8986\u76d6\u8865\u8bc1\u4e0e\u5bb9\u91cf\u4fee\u590d\u5206\u652f\uff1b\u975e\u65b0\u8bad\u7ec3": "Saved time-coverage evidence supplementation and capacity-repair branch; no new training",
    "\u4eba\u4f53\u6a21\u677f\uff0b\u836f\u7269\u8bc1\u636e\uff0b\u56db\u53c2\u6570\u6821\u51c6\uff0b\u65f6\u95f4\u8986\u76d6\u8865\u8bc1\u4e0e\u53ef\u884c\u6027\u4fee\u590d": "Human physiology, drug evidence, four-parameter calibration, time-coverage supplementation, and feasibility repair",
    "\u52a8\u8109\u3001\u9759\u8109\u4e0e\u95e8\u9759\u8109\u5faa\u73af": "Arterial, venous, and portal circulation",
    "15 \u4e2a\u5668\u5b98\u7684\u8840\u7ba1\u2014\u7ec4\u7ec7\u4ea4\u6362": "Vascular\u2013tissue exchange in 15 organs",
    "\u7ec4\u7ec7\u4e0e\u8840\u7ec6\u80de\u5206\u914d": "Tissue and blood-cell partitioning",
    "\u80be\u6e05\u9664\u53ca\u975e\u80be\u6e05\u9664": "Renal and nonrenal clearance",
    "\u9759\u8109\u7ed9\u836f\u4e0e\u8840\u6d46\u6d53\u5ea6\u8f93\u51fa": "Intravenous dosing and plasma concentration output",
    "\u975e\u80be\u6e05\u9664\u7387\u500d\u7387": "Nonrenal clearance multiplier",
    "\u500d": "fold",
    "\u7ec4\u7ec7\u5206\u914d\u500d\u7387": "Tissue partition multiplier",
    "\u53cc\u5411\u4ea4\u6362\u500d\u7387": "Bidirectional exchange multiplier",
    "\u8840\u7ec6\u80de\u5206\u914d\u500d\u7387": "Blood-cell partition multiplier",
    "\u5f53\u524d\u4ea4\u4e92\u8303\u56f4\u4e3a\u5df2\u6574\u7406\u7684 30 \u836f\u3001\u6210\u4eba\u3001\u5355\u836f\u3001\u5355\u6b21\u9759\u8109\u7ed9\u836f\uff1b\u9ed8\u8ba4\u4f7f\u7528\u6848\u4f8b\u7684\u9884\u6d4b\u6761\u4ef6\u3002": "Interactive scope is the 30 prepared drugs, adults, one drug, and a single intravenous dose; case prediction conditions are the default.",
    "\u65b0\u4eba\u4f53\u6761\u4ef6\u5fc5\u987b\u968f\u5bf9\u5e94\u5f00\u53d1\u89c2\u6d4b\u6570\u636e\u4e00\u8d77\u63d0\u4f9b\uff1b18\u201380 \u5c81\u4ec5\u4e3a\u4eba\u4f53\u6784\u5efa\u7684\u8f93\u5165\u8fb9\u754c\uff0c\u4e0d\u4ee3\u8868\u5df2\u6709\u6821\u51c6\u6216\u9a8c\u8bc1\u8986\u76d6\u3002": "New human conditions require corresponding development observations. Ages 18\u201380 are constructor input bounds, not established calibration or validation coverage.",
    "\u7ed3\u679c\u4e3a\u7814\u7a76\u6a21\u578b\u9884\u6d4b\uff0c\u6539\u53d8\u4eba\u4f53\u6216\u7ed9\u836f\u6761\u4ef6\u4e0d\u7b49\u4e8e\u5b8c\u6210\u4e86\u65b0\u573a\u666f\u7684\u72ec\u7acb\u79d1\u5b66\u9a8c\u8bc1\u3002": "Outputs are research-model predictions. Changing human or dosing conditions does not establish independent scientific validation of the new scenario.",
    "\u5f00\u53d1\u66f2\u7ebf\u6821\u51c6\u4e0e\u72ec\u7acb\u8bc4\u4ef7\u5206\u5f00\uff1b\u672a\u4e0a\u4f20\u5f00\u53d1\u6570\u636e\u65f6\u590d\u7528\u5df2\u4fdd\u5b58\u7684\u6821\u51c6\u53c2\u6570\u3002": "Development-curve calibration is separate from independent evaluation. Without new development data, saved calibrated parameters are reused.",
    "\u7ec8\u672b\u53c2\u6570\u4ec5\u5728\u6570\u503c\u8d28\u91cf\u68c0\u67e5\u901a\u8fc7\u65f6\u62a5\u544a\uff1b\u7f3a\u5c11\u72ec\u7acb\u4e34\u5e8a\u53c2\u8003\u7684\u7ec8\u70b9\u4e0d\u58f0\u79f0\u5df2\u9a8c\u8bc1\u3002": "Terminal parameters are reported only after numerical quality checks. Endpoints without independent clinical references are not claimed as validated."
}


def public_url(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    match = re.fullmatch(r"\[[^\]]*\]\((https?://.+)\)", value)
    if match:
        value = match[1].replace("\\&", "&")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if parsed.scheme not in {"https", "http"} or not host or parsed.username or parsed.password:
            return None
        if host.lower() in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            pass
        if any(re.search(r"token|api.?key|password|secret|signature|credential", key, re.I)
               for key, _ in parse_qsl(parsed.query)):
            return None
        return value
    except ValueError:
        return None


def pick(row, keys):
    return {key: row.get(key) for key in keys}


def content_id(digest):
    return "sha256:" + digest


@lru_cache(maxsize=32)
def verified_article_metadata(storage_path, checksum):
    """Recover article identifiers only from the registered, matching source."""
    if not storage_path or not re.fullmatch(r"[a-f0-9]{64}", str(checksum or "")):
        return {}
    path = Path(storage_path)
    if not path.is_absolute():
        path = ROOT / path
    # Old desktop paths are metadata, not permission to inspect another tree.
    if not path.is_relative_to(ROOT) or ".." in path.parts:
        return {}
    path = path.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        return {}
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != checksum:
        return {}
    if content.lstrip().startswith(b"<"):
        from lxml import etree
        try:
            root = etree.fromstring(content, etree.XMLParser(resolve_entities=False, no_network=True))
            articles = root.xpath("//PubmedArticle")
            result = {}
            for index, article in enumerate(articles, 1):
                pmid = article.findtext("MedlineCitation/PMID")
                title = "".join(article.find("MedlineCitation/Article/ArticleTitle").itertext()) if article.find("MedlineCitation/Article/ArticleTitle") is not None else None
                if pmid and pmid.isdigit():
                    result[str(index)] = {"pmid": pmid, "title": title}
            return {"pubmed_articles": result}
        except (etree.XMLSyntaxError, ValueError):
            return {}
    try:
        data = json.loads(content)
    except (ValueError, UnicodeDecodeError):
        return {}
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return {"fda_set_ids": {str(i): row.get("set_id") for i, row in enumerate(data["results"])
                                 if isinstance(row, dict) and row.get("set_id")}}
    return {}


def enrich_source_document(document):
    """Use exact singleton IDs or a hash-checked source locator; never guess."""
    doc = dict(document)
    url = public_url(doc.get("source_url"))
    metadata = dict((doc.get("parsed_record") or {}).get("metadata_json") or {})
    if url:
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        if parsed.hostname == "eutils.ncbi.nlm.nih.gov":
            ids = query.get("id", [])
            if len(ids) == 1:
                if query.get("db") == ["pmc"] and re.fullmatch(r"(?:PMC)?\d+", ids[0]):
                    metadata.setdefault("pmcid", "PMC" + ids[0].removeprefix("PMC"))
                if query.get("db") == ["pubmed"] and ids[0].isdigit():
                    metadata.setdefault("pmid", ids[0])
    location = str(doc.get("source_location") or "")
    if doc.get("source_id") == "pubmed" and not metadata.get("pmid"):
        match = re.search(r"xml_path=//PubmedArticle\[(\d+)\]/", location)
        if match:
            record = verified_article_metadata(doc.get("storage_path"), doc.get("checksum")).get("pubmed_articles", {}).get(match[1])
            if record:
                metadata["pmid"] = record["pmid"]
                doc["title"] = doc.get("title") or record.get("title")
    if doc.get("source_id") == "openfda_drug_label":
        match = re.search(r"\$\.results\[(\d+)\]", location)
        if match:
            set_id = verified_article_metadata(doc.get("storage_path"), doc.get("checksum")).get("fda_set_ids", {}).get(match[1])
            if set_id and re.fullmatch(r"[a-f0-9-]{36}", set_id, re.I):
                doc["source_url"] = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=" + set_id
    doc["parsed_record"] = {"metadata_json": metadata}
    return doc


def portable(value, key=""):
    """Keep scientific structure; replace local references by content IDs."""
    if isinstance(value, dict):
        if key in {"source_pins", "artifact_pins"}:
            return {content_id(digest): digest for digest in value.values()}
        result = {}
        for field, item in value.items():
            if re.search(r"password|secret|api_key|access_token|access_code", field, re.I):
                continue
            if field in {"path", "storage_path", "raw_storage_path", "parsed_storage_path"} and value.get("sha256"):
                result[field] = content_id(value["sha256"])
            elif field in {"source_url", "repository_url", "url"}:
                result[field] = public_url(item)
            else:
                result[field] = portable(item, field)
        return result
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if isinstance(value, str):
        value = ENGLISH_METADATA.get(value, value)
        if value.startswith(("/home/", "/Users/", "/Volumes/", "runtime/", "experiments/", "artifacts/", "data_process/", "src/", "scripts/")):
            return "export-reference:" + hashlib.sha256(value.encode()).hexdigest()
        # Scientific prose occasionally embeds a local path; do not publish it.
        value = re.sub(r"/(?:home|Users|Volumes)/[^\s\"'<>;,]+", "[project source]", value)
        return value
    return value


def database_snapshot():
    from sqlalchemy import text
    from pbpk_kb.api.evidence_routes import serialize_row
    from pbpk_kb.storage.db import SessionLocal
    from pbpk_kb.storage.final_evidence_query import serving_evidence_sql
    from pbpk_kb.web.provenance import article_links

    with SessionLocal() as db:
        db.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        compound_rows = db.execute(text("""SELECT compound_id::text, preferred_name AS compound_name,
            synonyms,pubchem_cid,chembl_id,inchi_key,cas_number FROM compound_identity
            ORDER BY lower(preferred_name),compound_id""")).mappings().all()
        evidence_rows = db.execute(text(serving_evidence_sql() + " ORDER BY e.evidence_id,e.attributed_compound_id")).mappings().all()
        # Exact stored article metadata, never the first PMID of a batch request.
        document_rows = db.execute(text("""SELECT e.evidence_id::text,e.attributed_compound_id::text AS compound_id,
            e.source_id,p.title,p.parsed_record,COALESCE(r.source_url,e.source_record->>'source_url') AS source_url,
            e.source_location,e.storage_path,e.checksum FROM auto_validated_evidence e
            LEFT JOIN parsed_document p ON p.parsed_document_id=e.parsed_document_id
            LEFT JOIN raw_document r ON r.raw_document_id=p.raw_document_id""")).mappings().all()
        documents = {(r["evidence_id"], r["compound_id"]): dict(r) for r in document_rows}
        counts = Counter(r["compound_id"] for r in evidence_rows)
        compounds = [serialize_row(dict(row)) | {"evidence_count": counts[row["compound_id"]]} for row in compound_rows]
        identities = {r["compound_id"]: r for r in compounds}
        evidence = []
        fields = ("evidence_id", "evidence_type", "compound_id", "compound_name", "parameter_name",
                  "numeric_value", "value_text", "unit", "canonical_value", "canonical_unit", "context",
                  "source_id", "source_name", "evidence_text", "curation_status", "species", "population", "route", "dose")
        for row in evidence_rows:
            doc = enrich_source_document(documents.get((row["evidence_id"], row["compound_id"]), {}))
            doc["source_id"] = row["source_id"]
            links = article_links(doc, identities.get(row["compound_id"]))
            url = next((public_url(x["url"]) for x in links if public_url(x["url"])), None)
            evidence.append(serialize_row(pick(row, fields)) | {"source_url": url, "source_title": doc.get("title")})
        model_rows = db.execute(text("""SELECT v.model_version_id::text,v.model_id::text,m.model_name,
            v.version_label,v.software,v.software_version,v.compounds,v.species,v.populations,v.routes,
            v.formulations,v.enzymes,v.transporters,v.compartments,v.process_types,v.output_names,
            v.repository_url,v.repository_commit,v.license_spdx_id,v.license_status,v.static_parse_status,
            v.execution_probe_status,v.reusable_candidate,v.simulation_count,v.parameter_record_count,
            v.component_record_count,v.process_record_count,v.report_count,v.dataset_reference_count,
            EXISTS(SELECT 1 FROM public_pbpk_model_validation_run r
              WHERE r.model_version_id=v.model_version_id AND r.scientific_accepted IS TRUE) AS scientific_accepted
            FROM public_pbpk_model_version v JOIN public_pbpk_model_registry m USING(model_id)
            ORDER BY lower(m.model_name),v.version_label,v.model_version_id""")).mappings().all()
        models = [serialize_row(dict(r)) for r in model_rows]
        rows = db.execute(text("""SELECT result.recommendation_result_id,result.compound_name,result.objective,
            result.overall_confidence,result.request_json,result.recommendation_json,result.created_at
            FROM recommendation_result result WHERE recommendation_model_run_id=(
                SELECT recommendation_model_run_id FROM recommendation_model_run
                WHERE engineering_acceptance IS TRUE ORDER BY created_at DESC,recommendation_model_run_id DESC LIMIT 1)
            AND result.status='recommendation_validated'
            ORDER BY lower(compound_name),objective,recommendation_result_id""")).mappings().all()
        evidence_by_id = {row["evidence_id"]: row for row in evidence}
        components = []
        for row in rows:
            result = row["recommendation_json"]
            citations = []
            for citation in result.get("citations", []):
                item = pick(citation, ("evidence_id", "evidence_type", "parameter", "source_id", "supporting_text", "confidence"))
                item["source_url"] = evidence_by_id.get(item["evidence_id"], {}).get("source_url")
                citations.append(item)
            components.append(serialize_row({
                "result_id": row["recommendation_result_id"], "compound_name": row["compound_name"],
                "objective": row["objective"], "request": row["request_json"], "saved_result": True,
                "overall_confidence": row["overall_confidence"], "created_at": row["created_at"],
                "component_recommendations": result.get("component_recommendations", []),
                "parameter_candidates": result.get("parameter_candidates", []),
                "recommended_model_structure": result.get("recommended_model_structure", {}),
                "evidence_gaps": result.get("evidence_gaps", []), "citations": citations,
            }))
    return dict(compounds=compounds, evidence=evidence, models=models, components=components)


def saved_recommendations():
    from pbpk_kb.web.recommendations import display_result

    result = []
    fields = ("candidate_id", "model_version_id", "name", "family_id", "version", "compounds", "score",
              "rank", "raw_rank", "rank_end", "display_rank", "display_rank_end", "display_layer",
              "decision_label", "reason_codes", "direct_reuse_allowed", "qualification_actions", "source_url")
    for path in sorted((ROOT / "runtime/deployment/workspace/recommendations").glob("*/result.json")):
        status_path = path.parent / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if status.get("status") != "complete":
            continue
        record = display_result(json.loads(path.read_text()))
        item = {"job_id": path.parent.name, "created_at": status.get("created_at"),
                "request": record.get("request", {}), "status": record.get("status"),
                "method": record.get("method"), "mode": "saved_result", "saved_result": True,
                "candidate_count": record.get("candidate_count"), "request_gaps": record.get("request_gaps", []),
                "limitations": record.get("limitations", []), "display_notice": record.get("display_notice")}
        for category in ("top_candidates", "reference_candidates", "blocked_candidates", "ranking"):
            item[category] = [pick(row, fields) for row in record.get(category, [])]
        result.append(item)
    return result


def referenced_urls(value, urls):
    if isinstance(value, dict):
        if value.get("sha256"):
            url = public_url(value.get("url") or value.get("source_url"))
            if url:
                urls[value["sha256"]] = url
        for item in value.values():
            referenced_urls(item, urls)
    elif isinstance(value, list):
        for item in value:
            referenced_urls(item, urls)


def blueprint_snapshot(stamp):
    from pbpk_kb.web.blueprints import _bundle, _catalog, _presentation, _path

    output = []
    cases = _catalog()[1]["cases"]
    for case in sorted(cases, key=lambda c: c["drug"].lower()):
        bundle = _bundle(case["case_id"])
        # _bundle audits source pins; explicitly audit exported artifact pins too.
        for path, expected in bundle["physiology"]["artifact_pins"].items():
            if hashlib.sha256(_path(path).read_bytes()).hexdigest() != expected:
                raise ValueError("Native artifact hash mismatch for " + case["case_id"])
        request = deepcopy(bundle["request"])
        request["partition_method"] = bundle["fitted"]["formula"]
        presentation = _presentation(bundle, request=request)
        urls = {}
        referenced_urls(case, urls)
        for index, row in enumerate(presentation["evidence"]):
            row["source_url"] = public_url(row.get("source_url")) or urls.get(request["evidence"][index]["source"]["sha256"])
        presentation["sources"] = []  # Single-evidence links are the UI provenance.
        # The complete curve is already stored once in saved_curve. Display
        # series remain in presentation.curve; avoid two further dense copies.
        presentation["endpoints"].pop("plasma_concentration_time", None)
        presentation["endpoint_summary"]["endpoints"].pop("plasma_concentration_time", None)
        calibration = presentation["calibration"]
        calibration["source_label"] = (bundle["training"].get("source_label") or bundle["training"].get("study_label")
                                        or calibration.get("study_id") or "Existing development study")
        output.append({
            "case_id": case["case_id"], "compound": case["drug"], "drug": case["drug"],
            "label": case["drug"], "request": request, "physiology": bundle["physiology"],
            "parameters": bundle["fitted"]["parameters"], "calibration": calibration,
            "saved_curve": bundle["saved_curve"], "evidence": presentation["evidence"],
            "training": bundle["training"], "presentation": presentation,
            "provenance": {"source_assets_verified_at_export": True, "source_assets_present_in_cloud": False,
                           "source_result_sha256": bundle["result_reference"]["sha256"],
                           "source_physiology_sha256": bundle["physiology_reference"]["sha256"],
                           "exported_at": stamp, "calibration_data_kind": "published_study_summary",
                           "individual_patient_records_included": False},
        })
        print(json.dumps({"blueprint_exported": case["drug"], "completed": len(output), "total": len(cases)}), flush=True)
    return {"schema_version": 1, "metadata": {"schema_version": 1, "exported_at": stamp, "source": "project_snapshot"}, "cases": output}


def main():
    sys.path.insert(0, str(ROOT / "scripts/deployment"))
    from service_runner import environment
    values = environment()
    os.environ.clear()
    os.environ.update(values)
    sys.path.insert(0, str(ROOT / "src"))
    stamp = datetime.now(timezone.utc).isoformat()
    catalog = database_snapshot()
    catalog["recommendations"] = saved_recommendations()
    blueprints = blueprint_snapshot(stamp)
    counts = {"evidence": len(catalog["evidence"]), "compounds": len(catalog["compounds"]),
              "covered_compounds": sum(bool(r["evidence_count"]) for r in catalog["compounds"]),
              "models": len(catalog["models"]), "families": len({r["model_id"] for r in catalog["models"]}),
              "blueprints": len(blueprints["cases"]), "components": len(catalog["components"]),
              "saved_recommendations": len(catalog["recommendations"])}
    catalog["metadata"] = {"schema_version": 1, "exported_at": stamp, "snapshot_at": stamp,
                           "source": "project_snapshot", "counts": counts,
                           "evidence_layer": "auto_validated", "database_snapshot": "repeatable_read_read_only",
                           "source_link_policy": "exact_stored_article_metadata; no inferred batch PMID"}
    payloads = {"catalog.json": portable(catalog), "blueprints.json": portable(blueprints)}
    encoded = {name: json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
               for name, value in payloads.items()}
    if sum(map(len, encoded.values())) > 25 * 1024 * 1024:
        raise ValueError("Cloud snapshot exceeds the 25 MiB publication limit")
    for content in encoded.values():
        if re.search(rb"/(?:home|Users|Volumes)/|(?i:password|access_token|access_code|api_key)\"\s*:", content):
            raise ValueError("Private local reference or credential key in cloud snapshot")
    manifest = {"schema_version": 1, "exported_at": stamp, "counts": counts,
                "files": {name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
                          for name, value in encoded.items()}}
    DEST.mkdir(parents=True, exist_ok=True)
    # Manifest last: consumers can detect interruption rather than use mixed data.
    encoded["manifest.json"] = json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    for name, content in encoded.items():
        temporary = DEST / (name + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(DEST / name)
    print(json.dumps({"status": "complete", "counts": counts, "files": manifest["files"]}), flush=True)


if __name__ == "__main__":
    main()
