#!/usr/bin/env python3
"""Audit original model sources for every new-study same-drug candidate.

Extracts only old model observed-dataset *names*, public source references and
new study identity metadata. Never reads new clinical outcome table values.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
import xml.etree.ElementTree as ET

from reviewer_source_overlap import ROOT, documents, file_ref, scan_documents, verify_ref

BASE = ROOT / "experiments/model_recommendation/reviewer_revision_20261009"
OUT = BASE / "external_validation"


def old_dataset_names(source):
    if source.suffix == ".json":
        d = json.loads(source.read_text())
        return [x.get("Name", "") for x in d.get("ObservedData", [])], "snapshot ObservedData[].Name"
    if source.suffix.casefold() == ".pksim5":
        with sqlite3.connect("file:" + str(source.resolve()) + "?mode=ro", uri=True) as db:
            names = [r[0] for r in db.execute("SELECT d.Name FROM DATA_REPOSITORIES d JOIN OBSERVED_DATA o ON o.DataRepositoryId=d.Id ORDER BY d.Name")]
        return names, "read-only SQLite OBSERVED_DATA join DATA_REPOSITORIES, Name only; no content or quantity columns"
    return [], "metadata format not supported; retained unknown"


def main():
    selection_path = OUT / "selection_manifest.json"
    selected = json.loads(selection_path.read_text())["selected_scenarios"]
    candidate_path = BASE / "source_overlap/candidate_study_roles.jsonl"
    old = [json.loads(line) for line in candidate_path.read_text().splitlines()]
    pairs = []
    for scene in selected:
        candidates = {r["candidate_id"]: r for r in old if r["compound"] == scene["compound"]}
        article = ET.parse(OUT / "sources" / (scene["pmcid"] + ".xml")).getroot()
        title = " ".join(article.find("front/article-meta/title-group/article-title").itertext())
        identifiers = {"doi": scene["doi"], "pmcid": scene["pmcid"], "title": title}
        if scene["pmcid"] == "PMC13624280":
            identifiers["trial"] = "NCT06084156"
        elif scene["pmcid"] == "PMC13104799":
            identifiers["trial"] = "NCT04211272"
        for cid, candidate in sorted(candidates.items()):
            source = ROOT / candidate["source_model"]["path"]
            verify_ref(candidate["source_model"])
            directory = next((p for p in source.parents if p.name == "source"), source.parent)
            docs, skipped = documents(directory)
            matches = scan_documents(docs, identifiers)
            names, extraction = old_dataset_names(source)
            years = [int(y) for name in names for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", name)]
            README = directory / "README.md"
            source_declaration = []
            if README.is_file():
                for i, line in enumerate(README.read_text().splitlines(), 1):
                    if re.search(r"clinical studies|observed data|model establish|all published|used for model|\[\d+\].*(?:doi|http)", line, re.I):
                        source_declaration.append({"source": file_ref(README), "line_1based": i, "text": line})
            pairs.append({
                "scenario_id": scene["scenario_id"], "candidate_id": cid, "compound": scene["compound"],
                "target_identifiers": identifiers, "source_model": candidate["source_model"],
                "status": "target_identity_match_review_required" if matches else "no_target_identity_found_in_reviewed_public_source_documents_or_names",
                "parameter_calibration_independence_proven": False,
                "new_clinical_study_independence_proven": False,
                "source_matches": matches, "source_declarations": source_declaration,
                "source_documents": list({d["source"]["path"]: d["source"] for d in docs}.values()), "documents_not_scanned": skipped,
                "registered_old_observed_dataset_names": names, "dataset_names_extraction": extraction,
                "registered_old_dataset_count": len(names), "latest_year_in_old_dataset_names": max(years) if years else None,
                "role_of_chronology": "Auxiliary only; publication/dataset years cannot rule out undocumented data reuse.",
                "limitations": ["This candidate-specific search describes reviewed public sources and embedded dataset identities only; it does not prove universal absence of undocumented study or participant overlap.",
                                "Clinical outcome values from the new papers were not inspected by this audit.",
                                "No automatic independence status is inherited from a different same-drug model or snapshot version."]})
    path = OUT / "candidate_source_audit.json"
    payload = {"schema": "reviewer-external-candidate-source-audit-v1", "selection": file_ref(selection_path),
               "old_candidate_bindings": file_ref(candidate_path), "tasks": len(pairs), "rows": pairs,
               "target_identity_matches": sum(bool(r["source_matches"]) for r in pairs),
               "independence_proven_tasks": 0, "no_outcome_selection": True,
               "claim_boundary": "New study exposure is separated from original-model-development independence; neither no-hit nor a later publication date establishes the latter."}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: payload[k] for k in ("tasks", "target_identity_matches", "independence_proven_tasks")}))


if __name__ == "__main__":
    main()
