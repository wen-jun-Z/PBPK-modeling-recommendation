#!/usr/bin/env python3
"""Seal published statin mean curves without exposing observations to design code.

Only the day -4 parent-drug rows of Supplemental Tables 1 and 4 are used.
This curator performs no prediction, fitting, result selection, or network access.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from pypdf import PdfReader
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "experiments/blueprint_design/independent_cohort_20260920"
AUDIT = ROOT / "experiments/blueprint_design/external_validation/concentration_curve_audit_20260920"
OUT = AUDIT / "aggregate_statin_curves"
PDF = AUDIT / "supplement_round2/sealed/PMC12378308/40256_2025_740_MOESM1_ESM.pdf"
ARTICLE = COHORT / "sealed/sources/PMC12378308.xml"
NUMBER = re.compile(r"\d+(?:\.\d+)?")
CELL = re.compile(r"(?:\d+(?:\.\d+)?|-)")
SPECS = (
    ("atorvastatin", 1, 1, 24, 42, 1),
    ("rosuvastatin", 7, 4, 22, 32, 2),
)


def ref(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def write(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if "sealed" in path.relative_to(OUT).parts:
        path.parent.chmod(0o700)
        path.chmod(0o600)
    return ref(path)


def parse_rows(reader, document, page_index: int, drug: str, count: int):
    text = reader.pages[page_index].extract_text()
    prefix, marker, _ = text.partition(drug.capitalize() + " +")
    assert marker, "Cannot locate monotherapy/combination boundary"
    rows = [line.split() for line in prefix.splitlines()
            if len(line.split()) == 5 and all(CELL.fullmatch(v) for v in line.split())]
    assert len(rows) == count, "Unexpected baseline table size"
    words = document[page_index].get_text("words")
    times = sorted((w for w in words if 210 < w[0] < 275 and w[1] > 114 and NUMBER.fullmatch(w[4])), key=lambda w: w[1])[:count]
    independent_rows = []
    for time in times:
        same_line = sorted((w for w in words if abs(w[1] - time[1]) < 1.5 and w[0] > 210 and CELL.fullmatch(w[4])), key=lambda w: w[0])
        assert len(same_line) == 5
        # Exact source columns: time, geometric mean, arithmetic mean, SD, CV.
        for word, bounds in zip(same_line, [(210, 275), (280, 385), (395, 495), (500, 605), (625, 750)]):
            assert bounds[0] < word[0] < bounds[1]
        independent_rows.append([w[4] for w in same_line])
    assert rows == independent_rows, "Independent PDF parsers disagree"
    footer = " ".join(reader.pages[page_index + 1].extract_text().split())
    start = footer.index("Lower limit of quantitation")
    footer = footer[start:]
    assert "0.1 ng/mL" in footer and "below this limit is reported as 0" in footer
    assert "Geometric mean was not reported if any zero values" in footer
    return rows, footer


def demographic_metadata(column: int, n: int) -> dict:
    table = next(x for x in ET.parse(ARTICLE).getroot().iter("table-wrap") if x.get("id") == "Tab1")
    cells = {}
    for row in table.iter("tr"):
        values = [" ".join("".join(c.itertext()).split()) for c in row if c.tag in {"td", "th"}]
        if len(values) == 3:
            cells[values[0]] = values[column]
    def count(label):
        return int(cells[label].split()[0])
    def mean_sd(label):
        match = re.fullmatch(r"([\d.]+) \(([\d.]+)\)", cells[label])
        assert match
        return {"mean": float(match[1]), "sd": float(match[2])}
    sex = {"female": count("Female"), "male": count("Male")}
    assert sum(sex.values()) == n
    return {
        "source": "PMC12378308 Table 1; full baseline cohort, not the smaller paired endpoint subset",
        "n": n,
        "sex_counts": sex,
        "sex_weights": {key: value / n for key, value in sex.items()},
        "age_year": mean_sd("Age, years, mean (SD)"),
        "body_weight_kg": mean_sd("Body weight, kg, mean (SD)"),
        "height_cm": mean_sd("Height, cm, mean (SD)"),
        "bmi_kg_m2": mean_sd("BMI, kg/m2, mean (SD)"),
        "race_counts": {label: count(label) for label in ["White", "Asian", "Black/African American", "Multi-racial"]},
        "ethnicity_counts": {label: count(label) for label in ["Hispanic or Latino", "Not Hispanic or Latino"]},
        "individual_covariates_or_correlations_available": False,
        "per_timepoint_sample_count_reported": False,
        "population_denominator_is_table_header_not_invented_per_time_sample_size": True,
    }


def curate() -> dict:
    if OUT.exists():
        raise SystemExit("Output already exists; do not overwrite a potentially frozen package")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    reader, document = PdfReader(PDF), pymupdf.open(PDF)
    assert len(reader.pages) == 10
    model_audit = COHORT / "model_paper_audit/audit.json"
    model_records = json.loads(model_audit.read_text())["records"]
    cases = []
    frozen_inputs = {str(p.relative_to(ROOT)): ref(p)["sha256"] for p in [PDF, ARTICLE, model_audit]}
    for drug, page_index, table_num, count, n, cohort_index in SPECS:
        previous = COHORT / f"active_protocols_v3/prediction_metadata/{drug}_pmc12378308_monotherapy_day_minus4.json"
        old = json.loads(previous.read_text())
        frozen_inputs[str(previous.relative_to(ROOT))] = ref(previous)["sha256"]
        rows, source_footnote = parse_rows(reader, document, page_index, drug, count)
        times = [float(row[0]) for row in rows]
        assert all(a < b for a, b in zip(times, times[1:]))
        assert len(times) == len(old["protocol"]["sampling_times_min"])
        assert all(abs(t * 60 - registered) < 1e-8 for t, registered in zip(times, old["protocol"]["sampling_times_min"]))
        assert times[0] == 0 and times[-1] == 120
        case_id = old["case_id"] + "_mean_curve"
        population = demographic_metadata(cohort_index, n)
        records = []
        for index, row in enumerate(rows):
            assert all(float(v) >= 0 for v in row if v != "-")
            records.append({
                "source_row_ordinal": index + 1,
                "source_page_1based": page_index + 1,
                "time_h": float(row[0]),
                "arithmetic_mean_ng_ml": float(row[2]),
                "geometric_mean_ng_ml": None if row[1] == "-" else float(row[1]),
                "sd_ng_ml": float(row[3]),
                "cv_percent": None if row[4] == "-" else float(row[4]),
                "source_tokens": row,
                "geometric_mean_missing_reason": "source_suppressed_if_any_zero" if row[1] == "-" else None,
                "published_arithmetic_mean_is_zero": float(row[2]) == 0,
                "zero_interpretation": "source_censoring_or_reported_precision_not_proof_of_true_zero" if float(row[2]) == 0 else None,
            })
        censoring = {
            "individual_assay_lloq_ng_ml": 0.1,
            "source_individual_blq_rule": "below LLOQ reported as zero before group summaries",
            "source_geometric_mean_rule": "not reported if any zero individual values at that time",
            "arithmetic_group_mean_may_be_below_individual_lloq": True,
            "apply_individual_lloq_filter_to_positive_group_mean": False,
            "individual_blq_count_or_censored_values_recoverable": False,
            "source_zero_rows_retained": True,
            "source_footnote": source_footnote,
        }
        sealed = {
            "schema_version": "pbpk-independent-aggregate-curves-sealed-v1",
            "case_id": case_id, "compound": drug, "identifiers": old["identifiers"],
            "publication_cluster_id": old["publication_cluster_id"],
            "trial_cluster_ids": old["trial_cluster_ids"], "cohort_id": old["cohort_id"],
            "source_pdf": ref(PDF), "source_article": ref(ARTICLE),
            "source_table": f"Supplemental Table {table_num}",
            "source_page_1based": page_index + 1,
            "population_n": n, "individual_records_available": False,
            "primary_aggregation": "arithmetic_mean_of_source_BLQ_zeroed_individual_concentrations",
            "concentration_unit": "ng/mL", "time_unit": "h",
            "censoring": censoring,
            "records": records,
            "combination_day12_and_metabolites_excluded": True,
            "source_values_must_not_enter_design_or_training": True,
        }
        sealed_ref = write(OUT / "sealed" / f"{case_id}.json", sealed)
        clinical_audit = copy.deepcopy(old["clinical_source_audit"])
        papers = [r for r in model_records if r.get("compound") == drug]
        clinical_audit["original_model_paper_audit"] = papers
        clinical_audit["original_model_paper_fulltext_audit_available"] = bool(papers)
        if not papers:
            clinical_audit["remaining_audit_limit"] = "Registered snapshots and repository documents checked; separate full original-model-paper audit has not been claimed."
        source_audit = {
            "schema_version": "pbpk-independent-aggregate-source-audit-v1",
            "case_id": case_id,
            "source_pdf": ref(PDF), "source_article": ref(ARTICLE),
            "prior_endpoint_protocol": ref(previous),
            "clinical_source_audit": clinical_audit,
            "evidence_exclusion": copy.deepcopy(old["evidence_exclusion"]),
            "publication_and_trial_indivisible_for_development_evaluation_split": True,
            "all_NCT06081166_arms_excluded_from_development": True,
            "separate_cohorts_same_publication_not_independent_publications": True,
        }
        audit_ref = write(OUT / "source_audits" / f"{case_id}.json", source_audit)
        safe = {
            "schema_version": "pbpk-independent-aggregate-curve-design-v1",
            "case_id": case_id, "compound": drug, "study_id": old["study_id"],
            "cohort_id": old["cohort_id"], "identifiers": old["identifiers"],
            "publication_cluster_id": old["publication_cluster_id"],
            "trial_cluster_ids": old["trial_cluster_ids"],
            "tier": "independent_clinical_aggregate_curve_candidate",
            "status": "source_curve_and_prediction_metadata_prepared_native_prediction_not_executed",
            "protocol": copy.deepcopy(old["protocol"]),
            "population_metadata": population,
            "sampling_times_h": times,
            "sampling_time_tokens_h": [row[0] for row in rows],
            "all_scheduled_time_positions_retained": True,
            "primary_aggregation": sealed["primary_aggregation"],
            "aggregation_preselected_before_any_model_predictions": True,
            "evaluation_target": {
                "matrix": "plasma", "analyte": "parent", "binding": "total",
                "concentration_unit": "ng/mL", "individual_records_available": False,
                "curve_type": "published_arithmetic_group_mean",
                "source_subject_n": n, "scheduled_time_count": count,
                "concentration_curve_source_ready": True,
                "geometric_mean_sd_cv_preserved_sealed_as_auxiliary": True,
                "AUC_of_mean_curve_is_not_published_mean_individual_NCA": True,
                "Cmax_of_mean_curve_is_not_published_mean_individual_Cmax": True,
            },
            "censoring": censoring,
            "prior_endpoint_protocol": ref(previous),
            "prior_paired_endpoint_n": old["n_evaluable"],
            "baseline_curve_n": n,
            "denominator_note": "Supplemental baseline curves use full monotherapy cohort; paired endpoint subset is smaller and remains a separate target. Per-time sample size not reported.",
            "clinical_source_audit": clinical_audit,
            "source_audit": audit_ref,
            "evidence_exclusion": copy.deepcopy(old["evidence_exclusion"]),
            "source_pdf": ref(PDF), "source_article": ref(ARTICLE),
            "sealed_observation_reference": sealed_ref,
            "prediction_contract": {
                "freeze_model_selection_and_protocol_before_unsealing": True,
                "denylist_entire_publication_trial_and_supplement": True,
                "predict_at_original_source_grid_no_time_shift": True,
                "primary_aggregation_must_not_switch_after_predictions": True,
                "reference_subject_prediction_allowed_only_as_registered_approximation": True,
                "group_mean_reference_mismatch_requires_explicit_label": True,
                "demographic_joint_distribution_and_individual_covariates_unknown": True,
                "assay_censoring_mismatch_must_be_declared_before_prediction": True,
                "individual_curve_scorer_must_not_be_reused_without_aggregate_contract": True,
                "log_error_only_for_strictly_positive_published_mean": True,
                "positive_group_mean_below_individual_lloq_is_not_excluded": True,
                "predicted_nonpositive_for_positive_target_is_failure_not_clipped": True,
                "published_zero_rows_retained_with_censoring_reason": True,
                "study_level_clustering_required": True,
                "single_publication_drug_bootstrap_ci_forbidden": True,
            },
            "scientific_results_accessed_by_algorithm": False,
            "run_ready": False,
            "strict_main_evaluated": False,
            "remaining_execution_requirements": [
                "Verify compound/formulation native adapter and match fed dose protocol.",
                "Freeze population/reference approximation and assay handling.",
                "Lock all selected predictions and a separately registered aggregate-curve scorer before unsealing.",
            ],
        }
        # Exclude the supplement and the whole publication, including source paths.
        for obj in [safe["evidence_exclusion"], source_audit["evidence_exclusion"]]:
            obj["exclude_entire_trial"] = True
            obj["curator_only_source_paths"] = [str(ARTICLE.relative_to(ROOT)), str(PDF.relative_to(ROOT))]
        audit_ref = write(OUT / "source_audits" / f"{case_id}.json", source_audit)
        safe["source_audit"] = audit_ref
        protocol_ref = write(OUT / "protocols" / f"{case_id}.json", safe)
        cases.append({"case_id": case_id, "compound": drug, "population_n": n,
                      "scheduled_time_count": count, "curve_type": "published_arithmetic_group_mean",
                      "protocol": protocol_ref, "source_audit": audit_ref,
                      "sealed_observations": sealed_ref, "run_ready": False})
    for path, sha in frozen_inputs.items():
        assert ref(ROOT / path)["sha256"] == sha, "Source or prior protocol changed"
    validation = {
        "status": "passed", "cases": 2, "source_time_positions": 46,
        "independent_pdf_text_and_coordinate_parsers_match_every_cell": True,
        "original_registered_sampling_grid_matches": True,
        "supplement_monotherapy_n_matches_article_demographic_n": True,
        "nonnegative_source_tokens_and_missing_markers_preserved": True,
        "baseline_not_combination_rows": True, "old_frozen_inputs_unchanged": True,
        "source_invariance_sha256": frozen_inputs,
        "no_solver_or_prediction_calls": True, "no_observed_values_printed": True,
    }
    validation_ref = write(OUT / "static_validation.json", validation)
    receipt = {
        "schema_version": "pbpk-independent-aggregate-curation-receipt-v1",
        "status": "source_curves_sealed_safe_protocols_prepared",
        "cases": cases, "unique_compounds": 2, "unique_publications": 1,
        "unique_trials": 1, "separate_cohorts": 2, "aggregate_curves": 2,
        "individual_curves": 0, "scheduled_time_positions": 46,
        "new_candidate_compounds_added": 0,
        "study_identifiers": {"pmcid": "PMC12378308", "pmid": "40569335", "doi": "10.1007/s40256-025-00740-1", "registry_id": "NCT06081166"},
        "primary_target": "published arithmetic group means, selected before model prediction",
        "prior_endpoint_protocols_unchanged": True,
        "source_values_remain_sealed": True,
        "native_or_method_runs": 0, "gpu_used": False,
        "static_validation": validation_ref,
    }
    write(OUT / "curation_receipt.json", receipt)
    document.close()
    return receipt


if __name__ == "__main__":
    result = curate()
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "source_time_positions": result["scheduled_time_positions"],
                      "receipt": ref(OUT / "curation_receipt.json")}, ensure_ascii=False))
