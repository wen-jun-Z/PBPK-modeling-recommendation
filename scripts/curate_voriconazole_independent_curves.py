"""Seal exact public supplement curves; emit only observation-free design metadata.

No fitting, scoring, digitization, imputation, or original input modification.
The published C1..C17/18 columns are linked, in order, to Methods' nominal
sampling schedule. Actual individual sample times/covariates are not published.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/blueprint_design/external_validation/concentration_curve_audit_20260920"
OLD = ROOT / "experiments/blueprint_design/external_validation/newstudy_bundle_20260920"
ARTICLE = ROOT / "experiments/blueprint_design/independent_cohort_20260920/retrieval_round1/sealed_articles/PMC9985189.xml"
DOCX = OUT / "sealed/40360_2023_652_MOESM1_ESM.docx"
EXPECTED_DOCX_SHA = "326fa0f2c2070f89b2c114cdadaccdcc1ae1089f6569d45cd6a972d95a9da6be"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def write(path: Path, payload: object, sealed: bool = False) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"Refusing to overwrite a different curated artifact: {path}")
    path.write_bytes(content)
    if sealed:
        path.chmod(0o600)
    return reference(path)


def reference(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def cell_text(cell: ET.Element) -> str:
    return "".join(cell.itertext()).strip()


def main() -> None:
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    assert reference(DOCX)["sha256"] == EXPECTED_DOCX_SHA
    with zipfile.ZipFile(DOCX) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    tables = document.findall(".//w:tbl", NS)
    assert len(tables) == 8, "Expected four tables each continued across two physical tables"
    protocols, counts, validation = [], [], []
    total_curves = total_samples = 0
    for dose in (4, 6):
        prior_path = OLD / "protocols" / f"voriconazole_li2023_reference_iv_{dose}mgkg.json"
        prior = json.loads(prior_path.read_text())
        times = prior["nominal_sampling_hours_from_observation_origin"]
        assert len(times) == (17 if dose == 4 else 18)
        assert times[0] == 0 and times[-1] == 48 and all(a < b for a, b in zip(times, times[1:]))
        records = []
        subject_links = {}
        for treatment_index, treatment in enumerate(("T", "R")):
            source_index = (0 if dose == 4 else 4) + 2 * treatment_index
            merged = {}
            for physical_table_index in (source_index, source_index + 1):
                table_rows = tables[physical_table_index].findall("w:tr", NS)
                header = [cell_text(c) for c in table_rows[0].findall("w:tc", NS)]
                assert header[:5] == ["SUB", "SEQ", "PER", "GRP", "TRT"]
                rows = table_rows[1:25]
                assert len(rows) == 24
                for row_index, row in enumerate(rows, 1):
                    cells = [cell_text(c) for c in row.findall("w:tc", NS)]
                    assert len(cells) == len(header)
                    subject, sequence, period, group, actual_treatment = cells[:5]
                    assert subject.isdigit() and actual_treatment == treatment
                    assert sequence in ("TR", "RT") and int(period) in (1, 2)
                    assert sequence[int(period) - 1] == treatment
                    identity = dict(subject_id=subject, sequence=sequence, period=int(period), randomization_group=group, treatment=treatment)
                    row_record = merged.setdefault(subject, {**identity, "cells": {}})
                    assert all(row_record[k] == v for k, v in identity.items())
                    for column_index, (column, token) in enumerate(zip(header[5:], cells[5:]), 5):
                        assert re.fullmatch(r"C\d+", column) and column not in row_record["cells"]
                        sample_index = int(column[1:]) - 1
                        value = None if token in ("BQL", "/") else float(token)
                        assert value is None or value >= 0
                        status = {"BQL": "left_censored_below_LLOQ", "/": "source_missing_unspecified"}.get(token, "quantified")
                        row_record["cells"][column] = {"source_token": token, "nominal_time_h": times[sample_index], "concentration_ng_ml": value, "observation_status": status, "source_physical_table_index_1based": physical_table_index + 1, "source_row_index_0based": row_index, "source_column_index_0based": column_index}
            assert len(merged) == 24
            for subject in sorted(merged):
                record = merged[subject]
                expected_columns = [f"C{i + 1}" for i in range(len(times))]
                assert set(record["cells"]) == set(expected_columns)
                cells = record.pop("cells")
                record["observations"] = [cells[key] for key in expected_columns]
                records.append(record)
                link = subject_links.setdefault(subject, {"subject_id": subject, "sequence": record["sequence"], "randomization_group": record["randomization_group"], "periods": {}})
                assert link["sequence"] == record["sequence"] and link["randomization_group"] == record["randomization_group"]
                link["periods"][treatment] = record["period"]
            assert sum(r["treatment"] == treatment for r in records) == 24
        for link in subject_links.values():
            assert set(link["periods"].values()) == {1, 2}
        assert sum(x["sequence"] == "TR" for x in subject_links.values()) == 12
        assert sum(x["sequence"] == "RT" for x in subject_links.values()) == 12
        case_id = f"voriconazole_li2023_reference_iv_{dose}mgkg_curves"
        sealed = write(OUT / "sealed" / f"{case_id}.json", {
            "schema_version": "pbpk-independent-concentration-curves-sealed-v1",
            "case_id": case_id, "compound": "voriconazole", "publication_cluster_id": "PMC9985189",
            "registry_id": "NCT05330000", "dose_group_mg_kg": dose,
            "source_docx": reference(DOCX), "source_article": reference(ARTICLE),
            "unit": "ng/ml", "matrix": "plasma", "analyte": "total parent voriconazole",
            "assay_LLOQ_ng_ml": 10.0,
            "source_LLOQ": {"value": 0.010, "unit": "microgram/ml", "location": "Bioanalysis"},
            "raw_BQL_preserved": True,
            "source_slash_missing_token_preserved": True,
            "source_BQL_convention": "Before Tmax: 0 for PK analysis; after Tmax: ND for PK analysis, 0 for descriptive statistics. No imputation performed in this artifact.",
            "time_column_mapping": "C1..Cn ordered one-to-one with predose plus ordered nominal sampling schedule in article Methods; actual per-subject times are unavailable.",
            "records": records,
            "test_and_reference_are_paired_periods_not_independent_studies": True,
        }, sealed=True)
        protocol = {
            "schema_version": "pbpk-independent-curve-design-v1", "case_id": case_id,
            "compound": "voriconazole", "study_id": prior["study_id"],
            "registry_id": "NCT05330000", "publication_cluster_id": "PMC9985189",
            "cohort_id": f"NCT05330000_{dose}mgkg", "source_identifiers": prior["source_identifiers"],
            "dose": prior["dose"], "food": prior["food"], "population": prior["population"],
            "matrix": prior["matrix"], "analyte": prior["analyte"],
            "sampling": {
                "time_h": times, "time_origin": "start of infusion in each period",
                "time_type": "published nominal, not actual individual sample time",
                "source_mapping": "Methods ordered sampling schedule, including predose, mapped one-to-one to supplementary columns C1..Cn",
                "source_column_labels": [f"C{i + 1}" for i in range(len(times))],
                "infusion_end_h": prior["dose"]["infusion_duration_min"] / 60,
                "rounding_preserved": "4mg/kg nominal 1.33h remains 1.33, not changed to 80/60; no observation time shifted.",
            },
            "crossover": {
                "design": "two treatments, two sequences, two periods; within-subject crossover",
                "washout_days": 7, "prediction_treatment": "R", "reference_name": "Vfend",
                "all_subject_links": list(subject_links.values()),
                "reference_periods_n": 24, "test_periods_n": 24,
                "test_periods_sealed_but_not_additional_independent_studies": True,
                "dose_groups_are_distinct_cohorts_in_one_study": True,
            },
            "demographic_linkage": {
                "individual_covariates_available": False,
                "published_demographics_are_group_level_only": True,
                "sex_age_weight_genotype_by_subject_id_not_reported": True,
                "per_subject_actual_dose_mg_unavailable": True,
                "allow_source_reference_population_approximation_if_frozen_before_prediction": True,
                "do_not_claim_personalized_individual_prediction": True,
            },
            "curve_evaluation_contract": {
                "reference_individual_curves_n": 24, "reference_nominal_observation_slots": len(times) * 24,
                "complete_published_grid_available": True, "no_digitization": True,
                "assay_LLOQ_ng_ml": 10.0,
                "BQL": "Preserved as left-censored; exact concentrations below LLOQ are unknown. Never coerce to a positive invented observation for log error.",
                "primary_estimand_requires_freeze": "Group/reference-population concentration prediction on original nominal schedule; individual heterogeneity is observed but individual covariates are absent.",
                "not_all_observation_slots_are_exact_quantified_values": True,
                "source_missing_tokens_preserved_without_imputation": True,
                "statistical_unit": "study/drug for generalization; paired subjects within dose cohort for uncertainty, never independent train/test by subject or treatment",
            },
            "prior_endpoint_protocol": reference(prior_path),
            "source_independence_audit": prior["source_independence_audit"],
            "sealed_observation_reference": sealed,
            "evidence_exclusion": {"entire_publication": "PMC9985189", "trial": "NCT05330000", "all_doses_treatments_subjects_periods": True},
            "prediction_freeze_required": True, "curve_data_ready": True, "run_ready": False,
            "strict_main_evaluated": False, "outcomes_exposed_to_designer": False,
            "limitations": ["Individual demographics/genotypes and actual sampling times are unavailable.", "No new target subjects, test formulation, or dose group may enter development or model selection.", "Source-only fixed-model external prediction is permitted after native protocol/population and scoring definitions are frozen; no new Dcal/Dsel split is created from this study."],
        }
        protocol_ref = write(OUT / "protocols" / f"{case_id}.json", protocol)
        protocols.append(protocol_ref)
        counts.append({"case_id": case_id, "subjects": 24, "reference_curves": 24, "time_points_per_curve": len(times), "reference_observation_slots": len(times) * 24})
        validation.append({"case_id": case_id, "24_unique_subjects": True, "all_source_cells_preserved": True, "two_physical_tables_joined_on_full_subject_identity": True, "source_sequence_period_treatment_consistent": True, "12_TR_12_RT": True, "all_published_sampling_columns_present": True, "no_values_in_prediction_metadata": True})
        total_curves += len(records)
        total_samples += sum(len(r["observations"]) for r in records)
    assert total_curves == 96 and total_samples == 1680
    receipt = {
        "schema_version": "pbpk-independent-curve-curation-receipt-v1",
        "status": "independent_source_numeric_curves_sealed",
        "compound_count": 1, "compound": "voriconazole", "study_count": 1,
        "registry_id": "NCT05330000", "dose_cohorts": 2, "subjects_total": 48,
        "reference_subject_curves": 48, "reference_observation_slots": 840,
        "all_treatments_curves": total_curves, "all_treatments_observation_slots": total_samples,
        "protocols": protocols, "case_counts": counts,
        "source_docx": reference(DOCX), "source_article": reference(ARTICLE),
        "source_values_are_curator_only": True, "solver_calls": 0, "gpu_used": False,
        "actual_individual_covariates_available": False,
        "primary_scoring_and_prediction_freeze_pending": True,
        "does_not_increase_unique_compound_or_publication_count": True,
    }
    write(OUT / "curation_receipt.json", receipt)
    write(OUT / "static_validation.json", {"passed": True, "cases": validation, "all_96_curves_exactly_source_preserved": True, "no_train_test_split_created_from_external_subjects": True})
    print(json.dumps({"status": receipt["status"], "studies": 1, "subjects": 48, "reference_curves": 48, "reference_observation_slots": 840, "protocols": protocols}, indent=2))


if __name__ == "__main__":
    main()
