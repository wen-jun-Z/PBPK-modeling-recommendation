#!/usr/bin/env python3
"""Freeze source-only clinical arm choices before PK scoring.

Only methods without table/figure nodes and table captions/headers are exported.
An earlier parser unexpectedly exposed nested tables for PMC12640539; that source
is explicitly nonblind and is not eligible for the strict external subset.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/model_recommendation/reviewer_revision_20261009"
OUT = BASE / "external_validation"


def ref(p):
    return {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def text(node):
    return " ".join(" ".join(node.itertext()).split()) if node is not None else ""


def methods_and_inventory(path):
    root = ET.parse(path).getroot()
    inventory = [{"id": w.get("id"), "label": text(w.find("label")), "caption": text(w.find("caption")),
                  "column_headers": text(w.find("table/thead"))} for w in root.findall(".//table-wrap")]
    sections = []
    for sec in root.findall("./body/sec"):
        if "method" not in text(sec.find("title")).casefold():
            continue
        for parent in sec.iter():
            for child in list(parent):
                if child.tag in {"table-wrap", "fig", "supplementary-material"}:
                    parent.remove(child)
        for item in sec.findall("sec"):
            sections.append({"id": item.get("id"), "title": text(item.find("title")), "text": text(item)})
    return {"source": ref(path), "tables_figures_supplements_excluded": True, "sections": sections, "table_inventory_headers_only": inventory}


def main():
    manifest_path = OUT / "selection_manifest.json"
    if manifest_path.exists():
        frozen = json.loads(manifest_path.read_text())
        for source in frozen["source_receipts"]:
            if ref(ROOT / source["path"])["sha256"] != source["sha256"]:
                raise ValueError("frozen clinical source changed")
        print("Selection already frozen and source hashes unchanged; no rewrite.")
        return
    sources = []
    for pmc in ["PMC13624280", "PMC13104799", "PMC12640539", "PMC12511804"]:
        source = OUT / "sources" / (pmc + ".xml")
        parsed = methods_and_inventory(source)
        (source.parent / (pmc + "_methods.json")).write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n")
        sources.append(ref(source))
    scenes = [
        {"scenario_id": "external:metformin_pmc13624280_day4_monotherapy", "pmcid": "PMC13624280", "doi": "10.1002/cpdd.70092", "compound": "metformin",
         "arm": "Period 1 metformin alone, Day 4", "dose_mg": 1000.0, "route": "oral", "interval_h": 12.0,
         "administration_times_h": [0, 12, 24, 36, 48, 60, 72], "last_dose_reference_time_h": 72.0, "endpoint_window_relative_h": [0.0, 12.0],
         "food": "fasted on Day 4 PK dose; previous doses with meals", "formulation": "commercial oral metformin; release formulation not explicit in Methods",
         "population": "healthy Chinese adults aged18-60,BMI18-28,weight>=50kg; enrolled22",
         "sampling_times_relative_h": [0, 0.5, 1, 1.5, 2, 4, 6, 8, 12],
         "planned_endpoint_definitions": ["Cmax_ss", "AUC_tau_12h", "Tmax_ss"],
         "source_section_ids": ["cpdd70092-sec-0030", "cpdd70092-sec-0050"], "potential_observation_table_id": "cpdd70092-tbl-0002",
         "eligibility": "selected_before_observation_values", "strict_outcome_blind_eligible": True,
         "limitations": ["Do not relabel repeated-dose peak as single-dose Cmax.", "The BID-to-12h interval and seven-dose schedule are explicit operationalizations of Methods; day4 evening dosing must occur after the12h pre-next-dose sample.", "Food changes across doses and unspecified release formulation must not be silently treated as exact source-model condition matches."]},
        {"scenario_id": "external:pregabalin_pmc12511804_armA80mg", "pmcid": "PMC12511804", "doi": "10.3389/fphar.2025.1692519", "compound": "pregabalin",
         "arm": "Treatment A: pregabalin alone", "dose_mg": 80.0, "route": "oral", "interval_h": None,
         "administration_times_h": [0.0], "last_dose_reference_time_h": 0.0, "endpoint_window_relative_h": [0.0, 36.0],
         "food": "fasted >=10h before and4h after dosing", "formulation": "oral capsule",
         "population": "healthy adults aged18-50,BMI18-27; planned24,table caption reports23",
         "sampling_times_relative_h": [0, 0.16, 0.33, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.25, 2.5, 3, 6, 12, 24, 36],
         "planned_endpoint_definitions": ["Cmax", "AUC_inf", "Tmax", "terminal_half_life"],
         "source_section_ids": ["s2-1", "s2-3", "s2-5"], "potential_observation_table_id": "T2",
         "eligibility": "selected_before_observation_values", "strict_outcome_blind_eligible": True,
         "limitations": ["Methods says17 time points but explicitly lists16 including pre-dose; preserve the explicit list and discrepancy.", "Do not equate AUC_last with fixed AUC_0_36 without a reported last quantifiable time.", "Published2025 but trial approvals2016; publication date alone does not establish source-model-development independence."]},
    ]
    for study, food, schedule in [("A", "fasted", [0, 0.17, 0.25, 0.33, 0.5, 0.67, 0.83, 1, 1.33, 1.67, 2, 2.33, 2.67, 3, 4, 5, 6, 8, 10, 12, 16, 24]),
                                  ("B", "fed high-fat meal30min before dose", [0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 6, 8, 12, 16, 24])]:
        scenes.append({"scenario_id": f"external:sildenafil_pmc12640539_study{study}_fct100mg", "pmcid": "PMC12640539", "doi": "10.1093/sexmed/qfaf096", "compound": "sildenafil",
                       "arm": f"Study{study}film-coated tablet reference", "dose_mg": 100.0, "route": "oral", "interval_h": None,
                       "administration_times_h": [0.0], "last_dose_reference_time_h": 0.0, "endpoint_window_relative_h": [0.0, 24.0],
                       "food": food, "formulation": "film-coated tablet with240mL water", "sampling_times_relative_h": schedule,
                       "planned_endpoint_definitions": ["Cmax", "AUC_inf", "Tmax", "terminal_half_life"],
                       "source_section_ids": ["sec6", "sec8"], "potential_observation_table_id": "TB1" if study == "A" else "TB2",
                       "eligibility": "exploratory_only_inadvertent_outcome_visibility_before_arm_lock", "strict_outcome_blind_eligible": False,
                       "limitations": ["Methods node contained nested outcome tables; their values were accidentally rendered before arm selection was frozen.", "No model predictions/errors were inspected and no outcome-based arm exclusion was made; nevertheless this source cannot be described as fully outcome-blind.", "Both tablet reference arms retained together; experimental oral-film arms excluded by formulation rule, not numerical outcomes."]})
    for drug, dose, end in [("rosuvastatin", 10.0, 72.0), ("sildenafil", 20.0, 16.0)]:
        scenes.append({"scenario_id": f"external:{drug}_pmc13104799_day_minus4", "pmcid": "PMC13104799", "doi": "10.1002/jcph.70185", "compound": drug,
                       "arm": "Day-4 substrate monotherapy before macitentan", "dose_mg": dose, "route": "oral", "interval_h": None,
                       "administration_times_h": [0.0], "last_dose_reference_time_h": 0.0, "endpoint_window_relative_h": [0.0, end],
                       "food": "fasted", "formulation": "Crestor film-coated tablet" if drug == "rosuvastatin" else "Revatio film-coated tablet",
                       "planned_endpoint_definitions": ["Cmax", "AUC_inf", "Tmax", "terminal_half_life"],
                       "source_section_ids": ["jcph70185-sec-0040", "jcph70185-sec-0050", "jcph70185-sec-0070"],
                       "potential_observation_table_id": "jcph70185-tbl-0003",
                       "eligibility": "exclude_strict_new_study_subset_prior_project_download_plan_identity", "strict_outcome_blind_eligible": False,
                       "limitations": ["PMCID appears in old pending-download metadata; no old processed text hit was found, but absence of historical exposure cannot be asserted.", "Dosing paragraph explicitly says sildenafil20mg; the same article's formulation sentence says2mg. This inconsistency must be retained.", "Figure1 and main-text sampling-window descriptions disagree (16vs24h for sildenafil,72vs96h for rosuvastatin); do not calculate fixed-window AUC until resolved."]})
    # All schedules are relative to the dosing occasion under evaluation.
    explicit_schedules = {
        "rosuvastatin": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10, 12, 16, 24, 48, 72],
        "sildenafil": [0, 1/3, 2/3, 1, 1.5, 2, 2.5, 4, 5, 6, 8, 10, 12, 16],
    }
    for scene in scenes:
        pmc = scene["pmcid"]
        if pmc == "PMC13104799":
            scene["sampling_times_relative_h"] = explicit_schedules[scene["compound"]]
            scene["population"] = "healthy men aged18-55,BMI18-30,weight>=50kg; nominal18 per part"
        if pmc == "PMC12640539":
            scene["population"] = "healthy men; StudyA age18-45,BMI18.5-30,n45; StudyB age18-55,BMI18.5-30,enrolled35"
        scene["selection_stratum"] = (
            "strict_new_study_pre_outcome_selection" if scene["strict_outcome_blind_eligible"] else
            "exploratory_prior_project_metadata_identity" if pmc == "PMC13104799" else
            "exploratory_inadvertently_outcome_exposed")
        lloq = (25.0 if pmc == "PMC12511804" else 1.0 if pmc == "PMC12640539" else
                0.02 if pmc == "PMC13104799" and scene["compound"] == "sildenafil" else
                0.1 if pmc == "PMC13104799" else None)
        scene["study_id"] = "doi:" + scene["doi"]
        scene["normalized_design"] = {
            "route": scene["route"], "dose": {"value": scene["dose_mg"], "unit": "mg", "formulation": scene["formulation"]},
            "regimen": {"dose_times_h": scene["administration_times_h"], "observation_origin_h": scene["last_dose_reference_time_h"], "interval_h": scene["interval_h"]},
            "sampling": {"times_h": scene["sampling_times_relative_h"], "time_basis": "hours_after_evaluation_dose", "lloq_ng_ml": lloq},
            "population": scene["population"], "food": scene["food"], "matrix": "plasma", "analyte": "parent",
            "endpoint_contracts": [{"endpoint": name, "window_h": scene["endpoint_window_relative_h"] if name == "AUC_tau_12h" else None,
                                    "statistic": "arithmetic_mean" if pmc != "PMC13104799" else "geometric_mean",
                                    "predicted_cmax_definition": "maximum_at_nominal_sampling_times" if name.startswith("Cmax") else None}
                                   for name in scene["planned_endpoint_definitions"]],
        }
        scene["methods_source_ref"] = {**ref(OUT / "sources" / (pmc + "_methods.json")), "section_ids": scene["source_section_ids"]}
        scene["blocking_reasons"] = []
        if pmc == "PMC13624280":
            scene["limitations"].append("Methods does not state plasma assay LLOQ; do not invent a quantification cutoff for terminal NCA.")
        if pmc == "PMC13104799":
            scene["blocking_reasons"].append("Fixed sampling-window AUC blocked by Figure1/text sampling-horizon conflict; Cmax and reported AUC_inf may be evaluated separately with nominal schedule provenance.")
        # Endpoint values remain locked; statistic names are based on Methods,
        # and must be checked against table notes after prediction lock.
        scene["statistic_qualification_status"] = "Methods-based; verify actual table statistic after prediction lock; do not silently compare median Tmax to mean Tmax"
    manifest = {"schema": "reviewer-external-clinical-selection-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "root_protocol": ref(BASE / "protocol.json"), "source_receipts": sources,
                "metadata_shortlist": ref(BASE / "source_overlap/discovery/shortlist.json"),
                "selection_rule": "Commercial oral parent-drug monotherapy arms with explicit dose/regimen; preserve all qualifying reference arms; no selection on PK errors.",
                "primary_method": "Frozen original seed17 PBPK-Match document-only, target-drug grouped heldout head; no component/error correction or fitting on new studies.",
                "selected_scenarios": scenes, "strict_selected_scenarios": [s["scenario_id"] for s in scenes if s["strict_outcome_blind_eligible"]],
                "strict_selected_studies": ["PMC13624280", "PMC12511804"],
                "strict_studies_outcome_values_inspected": False,
                "inadvertent_exposure": {"pmcid": "PMC12640539", "tables": ["TB1", "TB2"], "stage": "Methods inspection before arm lock", "cause": "Outcome tables nested under Methods/Statistical analysis", "disposition": "exploratory only", "predictions_seen_before_selection": False},
                "all_new_source_model_development_independence": "not_established; requires candidate-specific supplementary source review",
                "not_a_completed_validation": True,
                "no_new_observation_fitted_or_used_for_model_selection": True}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"strict_selected_scenarios": manifest["strict_selected_scenarios"], "manifest": ref(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
