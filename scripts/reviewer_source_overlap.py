#!/usr/bin/env python3
"""Candidate-specific study reuse audit; never infer independence from no hit.

Reads the frozen 177 same-parent execution tasks, source documents, and prior
hash-bound audits. The original ranking, observations, and results are not edited.
Only execution disposition is copied from the results; numerical errors are not
used to select the audited tasks or classify source provenance.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/model_recommendation/execution_validation_20261001"
OUT = ROOT / "experiments/model_recommendation/reviewer_revision_20261009/source_overlap"
ATOR_AUDIT = ROOT / "experiments/blueprint_design/independent_cohort_20260920/model_paper_audit/atorvastatin_supplemental/source_independence_audit.json"
MAX_DOCUMENT = 16 * 1024**2


def safe_path(path, root=ROOT):
    p = (root / path).resolve(strict=True)
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError("source outside project or not a file")
    return p


@lru_cache(None)
def file_ref(path):
    p = safe_path(path)
    with p.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(p.relative_to(ROOT)), "sha256": sha, "bytes": p.stat().st_size}


def verify_ref(ref):
    actual = file_ref(safe_path(ref["path"]))
    if actual["sha256"] != ref["sha256"]:
        raise ValueError("frozen reference hash mismatch: " + ref["path"])
    return actual


def iter_refs(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for v in value.values():
            yield from iter_refs(v)
    elif isinstance(value, list):
        for v in value:
            yield from iter_refs(v)


def normalized(text):
    return re.sub(r"\s+", "", str(text)).casefold()


def identifier_hits(text, identifiers):
    value = normalized(text)
    hits = []
    for key, token in identifiers.items():
        if key not in {"doi", "pmid", "pmcid", "trial", "title"} or not token:
            continue
        if key == "title":
            title = re.sub(r"[^a-z0-9]", "", str(token).casefold())
            if len(title) >= 40 and title in re.sub(r"[^a-z0-9]", "", str(text).casefold()):
                hits.append({"kind": key, "value": str(token)})
            continue
        needle = normalized(token)
        # A PMID embedded inside a longer number does not establish identity.
        pattern = (r"(?<!\d)" + re.escape(needle) + r"(?!\d)") if key == "pmid" else re.escape(needle)
        if re.search(pattern, value):
            hits.append({"kind": key, "value": str(token)})
    return hits


def section_role(lines, index):
    """A study's use must be stated in its own section, not elsewhere in a file."""
    start = index
    while start > 0 and not re.match(r"^\s*#{1,6}\s", lines[start]):
        start -= 1
    context = "\n".join(lines[start:index + 1]).casefold()
    if "used for model building (training data)" in context or "used for model calibration" in context:
        return "calibration", start
    if "used for model verification" in context or "used for model validation" in context:
        return "validation", start
    return "mentioned_only", index


@lru_cache(None)
def documents(source_dir):
    import pymupdf
    docs, skipped = [], []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in {".md", ".txt", ".pdf"}:
            continue
        p = safe_path(path)
        ref = file_ref(p)
        if p.stat().st_size > MAX_DOCUMENT:
            skipped.append({**ref, "reason": "document_exceeds_16MiB"})
            continue
        if p.suffix.casefold() == ".pdf":
            with pymupdf.open(p) as pdf:
                for page_index, page in enumerate(pdf):
                    text = page.get_text()
                    docs.append({"source": ref, "page_1based": page_index + 1, "lines": text.splitlines()})
        else:
            docs.append({"source": ref, "lines": p.read_text(errors="replace").splitlines()})
    return docs, skipped


def scan_documents(docs, identifiers):
    matches, aliases = [], set()
    for doc in docs:
        lines = doc["lines"]
        for index, line in enumerate(lines):
            # Also check adjacent PDF lines when a DOI is wrapped.
            hits = identifier_hits("\n".join(lines[index:index + 4]), identifiers)
            if not hits:
                continue
            match = {"source": doc["source"], "lines_1based": [index + 1, min(index + 4, len(lines))],
                     "text": "\n".join(lines[index:index + 4]), "identity_hits": hits,
                     "role": "mentioned_only", "identity_support": "exact_publication_or_trial_identifier"}
            if "page_1based" in doc:
                match["page_1based"] = doc["page_1based"]
            matches.append(match)
            # Do not associate an earlier neighbouring bibliography entry with
            # a DOI found later in this four-line search window.
            aliases_on_identity_line = re.findall(r"\*\*([^*]+)\*\*", line) if identifier_hits(line, identifiers) else []
            for alias in aliases_on_identity_line:
                if re.search(r"\b(?:19|20)\d{2}\b", alias):
                    aliases.add(alias)
    # Bibliography alias -> explicit local model-building/verification section.
    for doc in docs:
        for index, line in enumerate(doc["lines"]):
            for alias in sorted(aliases):
                if f"[{alias}]" not in line:
                    continue
                role, start = section_role(doc["lines"], index)
                matches.append({"source": doc["source"], "lines_1based": [start + 1, index + 1],
                                "text": "\n".join(doc["lines"][start:index + 1]),
                                "bibliography_alias": alias, "role": role,
                                "identity_support": "same_repository_alias_linked_to_exact_identifier"})
    return matches


def classify(matches, previous_status, supplemental_pass=False):
    roles = {m["role"] for m in matches}
    if "calibration" in roles:
        return "calibration", "documented_source_study_overlap"
    if "validation" in roles:
        return "validation", "documented_source_study_overlap"
    if matches:
        return "mentioned_only", "documented_mention_usage_not_established"
    if previous_status == "documented_source_study_overlap":
        # Should not lose the known positive even if its public text layout changes.
        return "validation", "documented_source_study_overlap"
    if previous_status == "scoped_source_audit_pass" or supplemental_pass:
        return "not_established", "no_overlap_identified_in_previously_audited_scope"
    return "not_established", "unknown"


def supplemental_ator_pass(audit, scenario):
    if scenario["compound"] != "atorvastatin":
        return None
    old = json.loads(ATOR_AUDIT.read_text())
    snapshot = old["registered_model"]["snapshot"]
    if (snapshot["sha256"] != audit["catalog_provenance"]["source_sha256"]
            or safe_path(snapshot["path"]) != safe_path(audit["catalog_provenance"]["source_path"])):
        return None
    identifiers = old["target_clinical_study"]["identifiers_and_title"]
    if normalized(identifiers["DOI"]) != normalized(scenario["publication_ids"]["doi"]):
        raise ValueError("atorvastatin supplemental target differs")
    if (old["status"] != "registered_model_source_audit_complete_no_clinical_overlap_found"
            or old["known_clinical_overlap"] or old["unresolved_target_identity_alias_matches"]):
        return None
    refs = [verify_ref(r) for r in iter_refs(old)]
    return {"audit": file_ref(ATOR_AUDIT), "verified_references": list({r["path"]: r for r in refs}.values()),
            "scope": "Exact atorvastatin source snapshot, complete registered report and six prior-model supplements; clinical source-level nonoverlap only.",
            "first_added_to_task_audit_here": True}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios_path = BASE / "inputs/scenarios.jsonl"
    audits_path = BASE / "inputs/candidate_audit.jsonl"
    results_path = BASE / "followup_20261003/primary/results.json"
    final_accounting_path = BASE / "followup_20261003/final_accounting.json"
    final_accounting = json.loads(final_accounting_path.read_text())
    final_by_pair = {(r["scenario_id"], r["candidate_id"]): r for r in final_accounting["all_original_pairs"]}
    publication_metadata_path = OUT / "target_publication_metadata.json"
    publication_metadata = {r["id"]: r for r in json.loads(publication_metadata_path.read_text())["result"]["resultList"]["result"]}
    scenarios = {r["scenario_id"]: r for r in map(json.loads, scenarios_path.read_text().splitlines())}
    audits = [r for r in map(json.loads, audits_path.read_text().splitlines()) if r["identity_match"]]
    results = json.loads(results_path.read_text())
    previous = {(r["scenario_id"], r["candidate_id"]): r for r in results["candidate_results"] if r["identity_match"]}
    assert len(scenarios) == 41 and len(audits) == 177 and len(previous) == 177
    assert set(previous) == set(final_by_pair)
    for branch in final_accounting["branches"]:
        verify_ref(branch["receipt"])
    rows, doc_inventory = [], {}
    for audit in audits:
        sid, cid = audit["scenario_id"], audit["candidate_id"]
        scenario, result = scenarios[sid], previous[sid, cid]
        source = safe_path(audit["catalog_provenance"]["source_path"])
        verify_ref({"path": str(source.relative_to(ROOT)), "sha256": audit["catalog_provenance"]["source_sha256"]})
        source_dir = next((p for p in source.parents if p.name == "source"), source.parent)
        docs, skipped = documents(source_dir)
        for doc in docs:
            doc_inventory[doc["source"]["path"]] = doc["source"]
        identifiers = {k: v for k, v in scenario["publication_ids"].items() if k in {"doi", "pmid", "pmcid"}}
        target_metadata = publication_metadata[identifiers["pmid"]]
        if normalized(target_metadata.get("doi", "")) != normalized(identifiers["doi"]):
            raise ValueError("target metadata DOI disagrees with frozen identity")
        identifiers["title"] = target_metadata["title"]
        hits = scan_documents(docs, identifiers)
        old = result["source_audit"]
        final = final_by_pair[sid, cid]
        if final["executed_in_main"] != (result["reference_diagnostic"]["execution_status"] == "succeeded"):
            raise ValueError("final accounting and primary status disagree")
        verified = [verify_ref(ref) for ref in old.get("verified_references", [])]
        extra = supplemental_ator_pass(audit, scenario)
        role, status = classify(hits, old["status"], extra is not None)
        rows.append({
            "scenario_id": sid, "case_id": scenario["case_id"], "candidate_id": cid,
            "compound": scenario["compound"], "family_id": audit["family_id"], "family_name": audit["family_name"],
            "target_publication": identifiers, "publication_cluster_id": scenario["publication_cluster_id"],
            "source_model": file_ref(source), "repository_commit": audit["catalog_provenance"].get("repository_commit"),
            "protocol": verify_ref(scenario["source_protocol"]),
            "original_model_study_role": role, "overlap_status": status,
            "original_model_parameter_calibration_overlap_established": role == "calibration",
            "original_model_study_use_overlap_established": role in {"calibration", "validation"},
            "reviewed_source_document_count": len({d["source"]["path"] for d in docs}),
            "reviewed_source_documents": sorted({d["source"]["path"] for d in docs}),
            "documents_not_scanned": skipped, "exact_identity_and_role_evidence": hits,
            "prior_source_audit_status": old["status"], "prior_verified_references": verified,
            "supplemental_prior_audit": extra,
            "native_execution_status": result["reference_diagnostic"]["execution_status"],
            "primary_execution_status": result["reference_diagnostic"]["execution_status"],
            "final_execution_status": "succeeded" if final["executed_any_branch"] else "not_run",
            "final_execution_scope": "primary" if final["executed_in_main"] else "registered_supplement" if final["executed_any_branch"] else "still_unexecuted",
            "successful_registered_supplements": [{"receipt": verify_ref(s["receipt"]), "variant": s.get("variant"),
                                                   "analysis_stratum": s.get("analysis_stratum"), "supplement_scenario_id": s.get("scenario_id")}
                                                  for s in final.get("successful_supplements", [])],
            "final_status_evidence": {"source": file_ref(final_accounting_path), "json_pointer": "/all_original_pairs", "identity_key": [sid, cid]},
            "reference_scientific_status": result["reference_diagnostic"]["scientific_status"],
            "new_blind_external_validation": False,
            "interpretation": (
                "Re-execution on a study explicitly used in original model validation; not a new-study prediction. Calibration use is not established."
                if role == "validation" else
                "No identity match in the reviewed documents; this does not establish study or participant independence. Retain any earlier audit only at its declared scope."
                if not hits else "Documented citation or use retained; no independent-study claim is inferred."),
        })
    assert len({(r["scenario_id"], r["candidate_id"]) for r in rows}) == 177
    with (OUT / "candidate_study_roles.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    columns = ["scenario_id", "candidate_id", "compound", "family_name", "original_model_study_role", "overlap_status",
               "native_execution_status", "primary_execution_status", "final_execution_status", "final_execution_scope", "reference_scientific_status", "reviewed_source_document_count", "new_blind_external_validation"]
    with (OUT / "candidate_study_roles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in rows)
    summary = {
        "schema": "reviewer-candidate-study-overlap-v1", "scenarios": 41, "tasks": len(rows),
        "unique_candidates": len({r["candidate_id"] for r in rows}), "compounds": len({r["compound"] for r in rows}),
        "role_counts": dict(Counter(r["original_model_study_role"] for r in rows)),
        "overlap_status_counts": dict(Counter(r["overlap_status"] for r in rows)),
        "prior_status_counts": dict(Counter(r["prior_source_audit_status"] for r in rows)),
        "native_execution_counts": dict(Counter(r["native_execution_status"] for r in rows)),
        "execution_by_overlap_status": {s: dict(Counter(r["native_execution_status"] for r in rows if r["overlap_status"] == s)) for s in sorted({r["overlap_status"] for r in rows})},
        "final_execution_counts": dict(Counter(r["final_execution_status"] for r in rows)),
        "final_execution_scope_counts": dict(Counter(r["final_execution_scope"] for r in rows)),
        "final_execution_by_overlap_status": {s: dict(Counter(r["final_execution_status"] for r in rows if r["overlap_status"] == s)) for s in sorted({r["overlap_status"] for r in rows})},
        "source_documents_reviewed": len(doc_inventory), "source_documents_by_suffix": dict(Counter(Path(p).suffix for p in doc_inventory)),
        "new_blind_external_validation_tasks": 0,
        "limitations": [
            "A missing identifier is not evidence that the target study was absent from model development.",
            "No calibration overlap was established; zero confirmed calibration is not zero true calibration overlap.",
            "Previously scoped source audits are retained only after exact source/protocol/document hash verification.",
            "The atorvastatin supplemental audit is matched to its exact candidate and target DOI, not propagated to other statins or candidate versions.",
            "All 177 tasks retained. Primary layer136 succeeded/41 not run; after27 registered supplementary recoveries163 succeeded/14 not run. Main and supplement metrics are not pooled.",
            "The 41 existing scenarios have participated in project development and cannot become blind external data through this source audit.",
            "Repository markdown/text and available PDF reports were searched; undocumented development material and participant reuse remain unknown.",
        ],
        "selection_based_on_prediction_error": False,
        "sources": [file_ref(p) for p in (scenarios_path, audits_path, results_path, final_accounting_path, publication_metadata_path, ATOR_AUDIT, Path(__file__))],
        "outputs": {"rows": file_ref(OUT / "candidate_study_roles.jsonl"), "csv": file_ref(OUT / "candidate_study_roles.csv")},
    }
    (OUT / "document_inventory.json").write_text(json.dumps(list(doc_inventory.values()), ensure_ascii=False, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["tasks", "unique_candidates", "overlap_status_counts", "source_documents_reviewed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
