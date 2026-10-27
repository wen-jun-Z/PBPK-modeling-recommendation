#!/usr/bin/env python3
"""Merge safe curation receipts, without opening outcomes or score files."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/blueprint_design"
COHORT = BASE / "independent_cohort_20260920"
AUDIT = BASE / "external_validation/concentration_curve_audit_20260920"


def load(path):
    return json.loads(path.read_text())


def ref(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def save(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    acquisition_path = AUDIT / "supplement_round2/acquisition.json"
    acquisition = load(acquisition_path)
    statin_path = AUDIT / "aggregate_statin_curves/curation_receipt.json"
    statins = load(statin_path)
    for case in statins["cases"]:
        assert ref(ROOT / case["protocol"]["path"])["sha256"] == case["protocol"]["sha256"]
    audit_path = AUDIT / "article_curve_availability_audit.json"
    audit = load(audit_path)
    conclusions = {
        "PMC13284695": "DOCX readable text contains inclusion/exclusion criteria; no Word numerical concentration tables or editable chart/embedded workbook. One image remains unqualified; no individual numeric curve claimed.",
        "PMC10897852": "Both DOCX tables inspected: participant inclusion/exclusion and study procedures/visits; no numerical concentration-time table.",
        "PMC5297975": "All seven DOCX tables inspected: demographics, metabolite PK endpoints/geometric ratios, adverse events. No numeric parent-drug concentration-time table.",
        "PMC7993260": "DOCX contains LC-MS/MS assay methods, no numerical curve table. Separate Figure S1 EPS was not acquired and remains unqualified.",
        "PMC12378308": "Exact parent-drug concentration-time group tables in supplement: day -4 atorvastatin N42 (24 times) and rosuvastatin N32 (22 times); arithmetic primary with geometric/SD/CV auxiliary. Individual records not published in these tables.",
        "PMC9929805": "Official supplementary package reached 2MiB per-package cap and was discarded without retry. Attachments remain unqualified; no claim that numeric data are absent.",
    }
    lookup = {r["pmcid"]: r for r in acquisition["records"]}
    for row in audit["articles"]:
        pmc = row["pmcid"]
        if pmc not in lookup:
            continue
        download = lookup[pmc]
        files = {f["name"]: f for f in download["files"] if f.get("status") == "downloaded"}
        for attachment in row.get("explicit_supplement_links_checked", []):
            name = attachment["source_href"]
            attachment["opened_for_this_audit"] = name in files
            if name in files:
                attachment["local_reference"] = files[name]
        row["supplement_audit_result"] = conclusions[pmc]
        row["supplement_acquisition_status"] = download["status"]
        row["curve_qualification"] = "curated_arithmetic_group_curves_source_ready_prediction_pending" if pmc == "PMC12378308" else "no_additional_numerical_parent_curve_qualified_in_bounded_supplement_audit"
        if pmc == "PMC12378308":
            row["curve_curation_receipt"] = ref(statin_path)
            row["distinct_cohorts_same_publication_and_trial"] = True
            row["individual_records_available"] = False
    remaining_path = BASE / "external_validation/remaining_cohort_20260920/curve_availability.json"
    known = {r["pmcid"] for r in audit["articles"]}
    for row in load(remaining_path)["records"]:
        if row["pmcid"] in known:
            continue
        audit["articles"].append({
            "pmcid": row["pmcid"], "compounds": [row["compound"]],
            "previous_availability_audit": ref(remaining_path),
            "main_table_type": row["main_table_type"],
            "explicit_supplement_links_checked": row.get("explicit_supplement_media", []),
            "supplement_read": row.get("supplement_read", False),
            "curve_qualification": "no_numeric_parent_curve_qualified_in_existing_curated_source",
            "qualification_scope": "Only inspected materials; not proof no further author-held data exist.",
        })
    audit["target_publications"] = len(audit["articles"])
    audit["target_drugs"] = len({drug for row in audit["articles"] for drug in row["compounds"]})
    audit["scope"] = "21 already curated drugs/20 clinical publications: existing article tables plus bounded, explicit official supplement acquisition. Numeric source curves sealed; no digitized plots or simulated targets."
    audit["supplement_round2"] = {
        "acquisition": ref(acquisition_path), "attempted_publications": len(acquisition["records"]),
        "download_bytes": acquisition["new_download_bytes"],
        "maximum_new_download_bytes": acquisition["maximum_new_download_bytes"],
        "acquired_documents": sum(len(r["files"]) for r in acquisition["records"]),
        "oversize_package_skips": sum(r["status"] == "document_package_limit_reached_skipped" for r in acquisition["records"]),
        "no_retries_or_active_downloader": True,
    }
    audit.setdefault("initial_supplement_acquisition_download_bytes", audit["new_download_total_bytes"])
    audit.setdefault("initial_retained_supplement_bytes", audit["retained_supplement_bytes"])
    audit["new_download_total_bytes"] = audit["initial_supplement_acquisition_download_bytes"] + acquisition["new_download_bytes"]
    audit["retained_supplement_bytes"] = audit["initial_retained_supplement_bytes"] + sum(f.get("bytes", 0) for row in acquisition["records"] for f in row["files"])
    assert audit["new_download_total_bytes"] < audit["maximum_new_download_bytes"]
    audit["curve_source_coverage"] = {
        "unique_compounds": 3, "unique_publications": 2, "unique_trials": 2,
        "drug_names": ["atorvastatin", "rosuvastatin", "voriconazole"],
        "individual_curves": {"compound": "voriconazole", "reference_curves": 48, "dose_cohorts": 2, "nominal_positions": 840},
        "aggregate_curves": {"compounds": ["atorvastatin", "rosuvastatin"], "curves": 2, "cohorts": 2, "publications": 1, "nominal_positions": 46},
        "curve_types_must_be_evaluated_separately": True,
        "new_compounds_added_to_overall_candidate_count": 0,
    }
    save(audit_path, audit)
    summary_path = COHORT / "acquisition_summary.json"
    summary = load(summary_path)
    coverage = summary["concentration_curve_coverage"]
    # Preserve the exact previously reported Vori execution record and its score seal.
    if "individual_curve_validation" not in coverage:
        coverage["individual_curve_validation"] = copy.deepcopy(coverage)
    coverage.update({
        "status": "three_drugs_have_sealed_source_curves_one_fixed_source_reference_evaluated",
        "unique_compounds": 3, "unique_studies": 2, "unique_publications": 2,
        "drug_names": ["atorvastatin", "rosuvastatin", "voriconazole"],
        "registry_ids": ["NCT05330000", "NCT06081166"],
        "dose_cohorts": 4,
        "reference_individual_curves": 48,
        "aggregate_curves": 2,
        "aggregate_nominal_observation_slots": 46,
        "subjects_field_scope": "48 Voriconazole individual reference records only; statin populations have group summaries and are not appended as individual curves",
        "curve_types_must_be_evaluated_separately": True,
        "new_protocol_artifacts_are_views_of_existing_cases_not_extra_drugs_or_studies": True,
        "source_ready": True,
        "prediction_and_scoring_freeze_pending": True,
        "pending_prediction_scope": "Two newly prepared statin arithmetic group curves; completed Vori fixed-source reference evaluation remains unchanged",
        "strict_main_evaluated": False,
        "availability_audit_reference": ref(audit_path),
        "aggregate_curve_preparation": {
            "receipt": ref(statin_path), "unique_compounds": 2, "publications": 1,
            "trials": 1, "separate_cohorts": 2,
            "source_time_positions": 46, "reference_individual_curves": 0,
            "population_n": {"atorvastatin": 42, "rosuvastatin": 32},
            "paired_endpoint_subset_n_kept_separate": {"atorvastatin": 40, "rosuvastatin": 31},
            "primary_target": "arithmetic_mean_of_source_BLQ_zeroed_individual_concentrations",
            "protocols": [case["protocol"] for case in statins["cases"]],
            "prediction_cases_completed": 0,
            "native_execution_and_aggregate_score_contract_pending": True,
        },
        "qualification": "Three drugs across two independent clinical publications/trials have exact numerical source curves. Voriconazole has 48 reference individual curves in two dose cohorts and completed fixed-source-reference evaluation. Atorvastatin and rosuvastatin supply separate cohort arithmetic group curves in one shared publication/trial, newly sealed and not yet predicted or scored. This is not three drugs with completed M0-M6 experiments, and not the target 12+60-drug experiment.",
    })
    coverage["protocols"] = coverage["individual_curve_validation"]["protocols"] + [case["protocol"] for case in statins["cases"]]
    assert summary["curation_coverage"]["independent_clinical_candidate_drugs"] == 21
    assert summary["curation_coverage"]["prepared_protocols"] == 23
    summary["latest_concentration_supplement_round"] = audit["supplement_round2"]
    dirs = [COHORT] + [BASE / "external_validation" / name for name in (
        "newstudy_bundle_20260920", "additional_native_cohort_20260920",
        "concentration_curve_audit_20260920", "remaining_cohort_20260920")]
    summary["current_cohort_bytes"] = sum(p.stat().st_size for d in dirs for p in d.rglob("*") if p.is_file())
    assert summary["current_cohort_bytes"] < summary["maximum_cohort_bytes"]
    save(summary_path, summary)
    print(json.dumps({"candidate_drugs": 21, "curve_source_drugs": 3,
                      "curve_source_publications": 2, "new_aggregate_curves": 2,
                      "acquisition_summary": ref(summary_path), "availability_audit": ref(audit_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
