#!/usr/bin/env python3
"""Audit the pinned OSP source report against the sealed clinical target identity.

Reads source documents and clinical identity metadata only; never predictions or
score files. No downloads, model execution, or edits to frozen curve protocols.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/blueprint_design/independent_cohort_20260920"
OUT = BASE / "model_paper_audit/atorvastatin_supplemental"
ASSET = ROOT / "artifacts/pbpk-model-assets/public_collection_20260909/Atorvastatin-Model__bb12c16aa567"
SOURCE = ASSET / "source"
CURVES = ROOT / "experiments/blueprint_design/external_validation/concentration_curve_audit_20260920/aggregate_statin_curves"


def ref(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def normal(text):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower())


def plain(element):
    return " ".join(" ".join(element.itertext()).split())


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    target_article = BASE / "sealed/sources/PMC12378308.xml"
    registry = BASE / "protocol_revision_20260920/curator_only/NCT06081166.json"
    target = ET.parse(target_article).getroot()
    meta = target.find(".//article-meta")
    title = plain(meta.find("title-group"))
    authors = [plain(x) for x in meta.findall("contrib-group/contrib/name/surname")]
    registration = json.loads(registry.read_text())["protocolSection"]
    assert registration["identificationModule"]["nctId"] == "NCT06081166"
    start = registration["statusModule"]["startDateStruct"]
    assert start == {"date": "2023-11-11", "type": "ACTUAL"}
    tokens = {
        "PMCID": "PMC12378308", "PMID": "40569335",
        "DOI": "10.1007/s40256-025-00740-1", "NCT": "NCT06081166",
        "sponsor_protocol_id": "TA-8995-11", "exact_publication_title": title,
    }
    aliases = ["obicetrapib", "TA-8995"]
    manifest = json.loads((ASSET / "asset_manifest.json").read_text())
    expected = {r["path"]: r["sha256"] for r in manifest["files"]}
    source_files = [SOURCE / "README.md"] + sorted((SOURCE / "Evaluation/Input/Content").glob("*.md"))
    source_files += [SOURCE / "Atorvastatin-Model.json", SOURCE / "Evaluation/Input/evaluation_plan.json"]
    checks, documents = [], []
    for p in source_files:
        file_ref = ref(p)
        assert expected[str(p.relative_to(SOURCE))] == file_ref["sha256"], "Pinned source modified"
        documents.append((file_ref, p.read_text()))
    morse_file = OUT / "morse2019.xml"
    morse = ET.parse(morse_file).getroot()
    documents.append((ref(morse_file), plain(morse)))
    supplement_records = []
    for p in sorted((OUT / "supplements").glob("*.pdf")):
        pdf = PdfReader(p)
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        assert len(text) > 100, "Insufficient supplement text extraction"
        documents.append((ref(p), text))
        supplement_records.append({"source": ref(p), "pages": len(pdf.pages), "text_characters_checked": len(text), "entire_text_checked": True})
    assert len(supplement_records) == 6
    for file_ref, text in documents:
        ntext = normal(text)
        checks.append({
            "document": file_ref,
            "target_identifier_matches": [label for label, token in tokens.items() if normal(token) in ntext],
            "target_perpetrator_or_protocol_alias_matches": [a for a in aliases if normal(a) in ntext],
            "target_author_surname_matches": [a for a in authors if re.search(r"\b" + re.escape(a) + r"\b", text, re.I)],
        })
    strong = [r for r in checks if r["target_identifier_matches"]]
    weak = [r for r in checks if r["target_perpetrator_or_protocol_alias_matches"] or r["target_author_surname_matches"]]
    # A new positive identity or alias stops automatic eligibility; no favorable default.
    no_overlap = not strong and not weak
    readme = (SOURCE / "README.md").read_text()
    numbered = re.findall(r"^\[(\d+)\]\s+(.+)$", readme, re.M)
    assert len(numbered) == 15
    source_refs = (SOURCE / "Evaluation/Input/Content/References.md").read_text()
    full_refs = re.findall(r"^\*\*([^*]+)\*\*\s+(.+)$", source_refs, re.M)
    snapshot = json.loads((SOURCE / "Atorvastatin-Model.json").read_text())
    observation_labels = [r["Name"] for r in snapshot["ObservedData"]]
    label_years = [int(re.search(r"\b(19\d{2}|20\d{2})\b", label)[1]) for label in observation_labels]
    assert len(observation_labels) == 23 and max(label_years) == 2022
    target_ref_nodes = target.findall(".//back/ref-list/ref")
    morse_refs = list(morse.iter("ref"))
    assert len(morse_refs) == 41
    target_reference_titles = [plain(x) for r in target_ref_nodes for x in r.iter("article-title")]
    target_titles_used_in_registered_report = [t for t in target_reference_titles if len(normal(t)) > 30 and normal(t) in normal(readme + source_refs)]
    prior_search = json.loads((OUT / "prior_models_search.json").read_text())
    prior = [r for r in prior_search["resultList"]["result"] if r.get("id") in {"27858342", "26116278"}]
    assert len(prior) == 2
    curve_protocols = [CURVES / "protocols" / f"{d}_pmc12378308_monotherapy_day_minus4_mean_curve.json" for d in ["atorvastatin", "rosuvastatin"]]
    frozen = json.loads((CURVES / "curation_receipt.json").read_text())
    for case in frozen["cases"]:
        assert ref(ROOT / case["protocol"]["path"])["sha256"] == case["protocol"]["sha256"]
    resource_dirs = [BASE] + [ROOT / "experiments/blueprint_design/external_validation" / name for name in (
        "newstudy_bundle_20260920", "additional_native_cohort_20260920",
        "concentration_curve_audit_20260920", "remaining_cohort_20260920")]
    cohort_bytes = sum(p.stat().st_size for directory in resource_dirs for p in directory.rglob("*") if p.is_file())
    assert cohort_bytes < 100 * 1024**2
    audit = {
        "schema_version": "pbpk-pinned-model-clinical-source-independence-audit-v1",
        "compound": "atorvastatin",
        "case_id": "atorvastatin_pmc12378308_monotherapy_day_minus4_mean_curve",
        "status": "registered_model_source_audit_complete_no_clinical_overlap_found" if no_overlap else "clinical_independence_withdrawn_pending_overlap_resolution",
        "clinical_source_independence_supported": no_overlap,
        "eligibility_withdrawn": not no_overlap,
        "registered_model": {
            "repository": manifest["repository"], "commit": manifest["commit"],
            "source_url": manifest["source_url"], "asset_manifest": ref(ASSET / "asset_manifest.json"),
            "snapshot": ref(SOURCE / "Atorvastatin-Model.json"),
            "authoritative_original_model_source_type": "complete_OSP_model_building_and_evaluation_report",
            "report_title": "Building and Evaluation of a PBPK Model for Atorvastatin Acid in Healthy Adults",
            "standalone_journal_paper_identified_by_registered_repository": False,
            "report_entire_body_and_bibliography_checked": True,
            "report_and_readme_markdown_documents": len(source_files) - 2,
            "all_pinned_source_hashes_match_acquisition_manifest": True,
            "reference_bibliography_entries": len(full_refs),
            "clinical_publications_listed": len(numbered),
            "clinical_reference_bibliography": [{"number": int(k), "source_citation": v} for k, v in numbered],
            "observed_dataset_labels": observation_labels,
            "observed_dataset_count": len(observation_labels),
            "latest_observed_study_year": max(label_years),
            "source_clinical_label_years_precede_target_trial_start": max(label_years) < int(start["date"][:4]),
            "report_citation_inconsistencies_retained_not_silently_repaired": [
                "Mazzu source README/reference text has 2008; data section contains Mazzu2020, while actual snapshot labels the study Mazzu2000.",
                "Whitfield appears as 2010 and 2011; both antecede the target trial and were treated as aliases of the same cited study.",
            ],
        },
        "target_clinical_study": {
            "identifiers_and_title": tokens, "target_author_surnames": authors,
            "target_article": ref(target_article), "registry_metadata": ref(registry),
            "trial_actual_start": start, "publication_year": 2025,
            "target_article_full_text_and_reference_list_checked": True,
            "reference_count": len(target_ref_nodes),
            "target_reference_titles_found_in_registered_report": target_titles_used_in_registered_report,
            "curve_protocols_frozen_unchanged": [ref(p) for p in curve_protocols],
        },
        "prior_parameter_model_publications": {
            "not_identical_to_registered_OSP_model": True,
            "morse2019": {
                "identifiers": {"pmcid": "PMC6765700", "doi": "10.1002/psp4.12447"},
                "source": ref(morse_file), "platform": "Simcyp version17",
                "role_in_registered_OSP_report": "cited physicochemical solubility prior and mechanistic context",
                "entire_body_and_all_reference_entries_checked": True,
                "reference_count": len(morse_refs), "supplements": supplement_records,
                "model_binary_zip_not_executed_or_used_by_registered_snapshot": True,
                "publication_date": "2019-08-01",
                "publication_precedes_actual_target_trial_start": True,
            },
            "zhang2015_and_duan2017": [{
                "pmid": x["id"], "doi": x["doi"], "title": x["title"],
                "first_publication_date": x["firstPublicationDate"],
                "role": "historical physicochemical parameter citation, not registered OSP model identity",
                "full_text_review_claimed": False,
                "chronologically_cannot_report_2023_2024_target_trial_results": x["firstPublicationDate"] < start["date"],
            } for x in prior],
        },
        "document_level_identity_checks": checks,
        "known_clinical_overlap": strong,
        "unresolved_target_identity_alias_matches": weak,
        "eligibility_scope": {
            "clinical_trial_study_and_cohort_separation": True,
            "baseline_physiology_or_older_invitro_reference_sharing_is_not_clinical_overlap": True,
            "individual_participant_reidentification_not_available_or_required": True,
            "target_trial_and_all_its_arms_must_be_denied_to_development_retrieval_prompts_and_selection": True,
            "source_reference_audit_only_not_native_or_predictive_validation": True,
            "this_sidecar_supersedes_only_prior_missing_original_model_source_audit_note": True,
            "frozen_protocol_contents_unchanged": True,
            "no_new_target_drug_or_study_added": True,
        },
        "resource_and_execution": {
            "acquisition_events": ref(OUT / "download_events.json"),
            "cohort_bytes_measured_before_audit_write": cohort_bytes,
            "maximum_cohort_bytes": 100 * 1024**2,
            "optional_github_commit_timestamp_request": "403_rate_limit_stopped_without_retry; fixed commit/source hashes already established locally",
            "target_concentration_values_used": False,
            "project_experiment_predictions_or_scores_read": False,
            "solver_calls": 0, "gpu_used": False,
        },
    }
    result = OUT / "source_independence_audit.json"
    result.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": audit["status"], "reference_bibliography_entries": len(full_refs),
                      "registered_clinical_studies": len(numbered), "morse_reference_count": len(morse_refs),
                      "supplement_documents": len(supplement_records), "audit": ref(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
