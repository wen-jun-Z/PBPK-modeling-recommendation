#!/usr/bin/env python3
"""Make an outcome-blind external-study shortlist from metadata-only receipts.

This is discovery, not external validation or a claim of clinical independence.
Do not extract clinical numerical outcomes in this program. All candidates are
selected by title, public accessibility, and pre-existing project identity hits.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/model_recommendation/reviewer_revision_20261009/source_overlap/discovery"
STUDIES = [
    ("42811461", ["metformin"], "Cofrogliptin interaction study; confirm availability of metformin-alone arm and matching dose/regimen from Methods."),
    ("42023995", ["rosuvastatin", "sildenafil"], "Macitentan interaction study; use separately confirmed probe-alone periods. The two drugs share one publication cluster."),
    ("41287751", ["sildenafil"], "Oral-film food-effect crossover studies; confirm tablet reference arm, preparation, dose and food status before retaining."),
    ("41079751", ["pregabalin"], "Pregabalin/thioctic-acid fixed-dose combination; retain only if a scientifically comparable reference arm can be defined. No unmodeled combination is silently treated as monotherapy."),
    ("41804659", ["atorvastatin"], "Excluded: source identity already appears in project parsed documents and the old cohort inventory."),
    ("42416312", ["atorvastatin"], "Excluded: source identity already appears in project parsed documents and the old cohort inventory."),
]


def ref(p):
    return {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    metadata, sources = {}, []
    for p in sorted(OUT.glob("*_metadata.json")):
        receipt = json.loads(p.read_text())
        assert receipt["abstracts_or_outcomes_requested"] is False
        sources.append(ref(p))
        for r in receipt["result"]["resultList"]["result"]:
            metadata[r["id"]] = r
    hit_path = OUT / "preexisting_identity_hits.txt"
    hits = defaultdict(dict)
    for line in hit_path.read_text().splitlines():
        path, number, token = line.rsplit(":", 2)
        hits[token.casefold()].setdefault(path, int(number))
    audited = [json.loads(line) for line in (OUT.parent / "candidate_study_roles.jsonl").read_text().splitlines()]
    rows = []
    for pmid, drugs, note in STUDIES:
        record = metadata[pmid]
        identity_hits = []
        for token in (record.get("pmcid", ""), record.get("doi", "")):
            for path, line in sorted(hits.get(token.casefold(), {}).items()):
                identity_hits.append({"matched_identifier": token, "path": path, "line_1based": line})
        models = {r["candidate_id"]: {"candidate_id": r["candidate_id"], "compound": r["compound"],
                                      "family_name": r["family_name"], "source_model": r["source_model"]}
                  for r in audited if r["compound"] in drugs and r["native_execution_status"] == "succeeded"}
        rows.append({
            "pmid": pmid, "pmcid": record.get("pmcid"), "doi": record.get("doi"), "title": record["title"],
            "first_publication_date": record.get("firstPublicationDate"), "publication_year": record.get("pubYear"),
            "target_drugs": drugs, "europepmc_open_access_flag": record.get("isOpenAccess"),
            "public_fulltext_url": "https://pmc.ncbi.nlm.nih.gov/articles/" + record["pmcid"] + "/" if record.get("pmcid") else None,
            "metadata_url": "https://pubmed.ncbi.nlm.nih.gov/" + pmid + "/",
            "metadata_screening_disposition": "exclude_previous_project_exposure" if identity_hits else "provisional_external_study_candidate",
            "previous_project_identity_hits": identity_hits,
            "previously_executed_model_candidates": list(models.values()),
            "clinical_outcome_values_extracted": False,
            "cmax_auc_absolute_observation_availability": "not_inspected_in_discovery; must be established after protocol freeze",
            "study_or_participant_independence_proven": False,
            "selection_uses_prediction_error": False,
            "required_eligibility_checks": note,
        })
    # This audit deliberately excludes newly created reviewer revision artifacts.
    scan = {"roots": ["artifacts", "data_process", "experiments", "results"],
            "extensions": ["json", "jsonl", "md", "xml", "txt", "csv"],
            "excluded_path_components": ["reviewer_revision_20261009", "native", "migrations*", "engineering*", "cache", "fits", "work"],
            "identifiers": [x[k] for x in rows for k in ("doi", "pmcid") if x.get(k)],
            "scope": "Case-insensitive exact DOI/PMCID text scan; no-match does not prove absence from all undocumented development or model pretraining.",
            "raw_hit_count": len(hit_path.read_text().splitlines()), "source_receipt": ref(hit_path)}
    summary = {"schema": "reviewer-outcome-blind-study-discovery-v1", "requested_outcomes": False,
               "provisional_studies": sum(not r["previous_project_identity_hits"] for r in rows),
               "provisional_drugs": sorted({d for r in rows if not r["previous_project_identity_hits"] for d in r["target_drugs"]}),
               "excluded_previously_exposed_studies": sum(bool(r["previous_project_identity_hits"]) for r in rows),
               "studies": rows, "identity_scan": scan, "metadata_receipts": sources,
               "limitations": ["Titles and public metadata establish discovery eligibility only, not matching study conditions or usable numerical endpoints.",
                               "No participant outcome, AUC, Cmax or model prediction error was used to select studies.",
                               "Any subsequently rejected source must remain recorded with its nonperformance reason.",
                               "New external evaluation must use the frozen method and fixed candidate pool, not the winner of current ablations."]}
    (OUT / "shortlist.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("provisional_studies", "provisional_drugs", "excluded_previously_exposed_studies")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
