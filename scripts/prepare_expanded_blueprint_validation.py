#!/usr/bin/env python3
"""Curate preselected clinical IV metadata without exporting observation values.

The snapshot is used only for observations, dosing and human metadata. Its
Compound parameters, model fits, simulations and prediction outputs are never
used as construction evidence. Independent chemistry is supplied separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
VERSION = "expanded-clinical-cohort-1"
OUTPUT = "experiments/blueprint_design/expanded_validation_20261007/cohort.json"
INVENTORY = "experiments/blueprint_design/development_inputs/cohort_screening_20260918/source_inventory.json"
OLD_PROTOCOL = "experiments/blueprint_design/clinical_pbpk_comparison_20261004/protocol.json"

# Choices depend on population/route/source identity, never on concentrations
# or model performance. All twenty screened compounds remain in the artifact.
CHOICES = {
    "Amikacin": "Walker 1979 - Volunteer 9",
    "Clarithromycin": "Chu 1991, 250 mg iv",
    "Ethinylestradiol": "Ethinylestradiol.Back1981.0.03.iv.Mean",
    "Finerenone": "F A.Finerenone.Finerenone 0.25mg i.v..arith Mean",
    "Fluvoxamine": "Japanese Society 2015.30 mg iv",
    "Griseofulvin": "IV_99mg_Sub_D_Rowland",
    "Levonorgestrel": "Cicali 2021_Levonorgestrel_IV_0.09 mg",
    "Lisinopril": "Human (Swedish) IV Bolus 2.",
    "Mexiletine": "Campbell1978.Mexiletine.m.200.iv",
    "Moclobemide": "Raaflaub1984.Moclobemide.50.iv_inf_2min.sd.PatID.1",
    "Montelukast": "Cheng 1996 - 18mg IV male",
    "Oxycodone": "Nieminen 2009 OCD 0.1mg/kg",
    "Paclitaxel": "#102_Huizing-1993_Fig.4_135mg_m2_3h.",
    "Paroxetine": "Lund (1982) - paroxetine hydrochloride, 23 mg, iv, n=1 (EM), D, paroxetine",
    "Pitavastatin": "Kimoto et al 2022 - Pitavastatin 2 mg iv infusion",
    "Probenecid": "Dayton 1963 - 1860 mg IV Subject A",
    "Propofol": "Gepts 1987 - ID 202",
    "Rifampicin": "sanofi-aventis U.S. LLC. 2013 - 300 mg",
    "Rosuvastatin": "Martin 2003c - IV 8 mg control",
    "Vancomycin": "Boekh 1988 - mean 1g",
}
PENDING = {
    "Finerenone": ["Clinical study F A lacks a publicly identified primary study; internal workbook citation is insufficient"],
    "Fluvoxamine": ["Observation name is 30 mg but dosing protocol is 21.98 mg; salt/base conversion needs primary verification"],
    "Griseofulvin": ["ObservedData does not establish species or sample matrix; source manuscript required"],
    "Levonorgestrel": ["Cicali source identity and ng/L column-name versus mg/L storage-unit conversion need verification"],
    "Lisinopril": ["ObservedData does not establish species or sample matrix; source manuscript required"],
    "Moclobemide": ["Observation and protocol names say 2-min infusion, but actual protocol is a bolus"],
    "Oxycodone": ["Nieminen 2009 is a placebo/rifampin crossover; the archived series does not identify the selected arm, and 0.1 mg/kg oxycodone hydrochloride has no verified parent-dose conversion (PMID 19417618)"],
    "Paclitaxel": ["Observed population is female but individual template is male; mg/m2 dose uses unverified body surface area"],
    "Propofol": ["Actual QuantityInfo identifies arterial whole blood, not total venous plasma"],
}
PRIMARY_LINKS = {
    "Clarithromycin": "https://pubmed.ncbi.nlm.nih.gov/1387301/",
    "Ethinylestradiol": "https://pubmed.ncbi.nlm.nih.gov/6786829/",
    "Mexiletine": "https://www.open-systems-pharmacology.org/OSP-PBPK-Model-Library/Mexiletine/",
    "Oxycodone": "https://pubmed.ncbi.nlm.nih.gov/19417618/",
    "Paroxetine": "https://doi.org/10.1111/j.1600-0773.1982.tb01036.x",
}
STUDY_NAMES = {
    "Clarithromycin": "Chu 1991/1992 source-label family",
    "Oxycodone": "Nieminen 2009",
    "Paroxetine": "Lund 1982",
    "Levonorgestrel": "Cicali 2021",
    "Griseofulvin": "Rowland, year unverified",
    "Lisinopril": "Swedish clinical study, identity unverified",
}
MASS_UNITS = {"mg/l", "ug/l", "ug/ml", "ng/ml", "ng/l"}
MOLAR_UNITS = {"umol/l", "nmol/l", "mol/l"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, relative):
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Paths must stay inside this project")
    return path


def unit(value):
    return str(value).lower().replace("µ", "u").replace("μ", "u").replace(" ", "")


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def named(records, name):
    found = [(i, r) for i, r in enumerate(records) if r.get("Name") == name]
    if len(found) != 1:
        raise ValueError("Exactly one source record is required: " + name)
    return found[0]


def observation_metadata(observation, *, expected_drug):
    """Validate real paired observations; export only times/counts, never values."""
    props = {p["Name"]: p.get("Value") for p in observation.get("ExtendedProperties", [])}
    columns = observation.get("Columns", [])
    if len(columns) != 1:
        raise ValueError("Ambiguous primary concentration column")
    column = columns[0]
    grid = observation.get("BaseGrid", {})
    times, values = grid.get("Values", []), column.get("Values", [])
    tu, cu = unit(grid.get("Unit")), unit(column.get("Unit"))
    if tu not in {"h", "min"} or cu not in MASS_UNITS | MOLAR_UNITS:
        raise ValueError("Unsupported explicit observation units")
    if (len(times) != len(values) or not times or not all(finite(x) for x in times + values)
            or any(t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:]))
            or any(v < 0 for v in values)):
        raise ValueError("Finite nonnegative observations at increasing distinct times required")
    npositive = sum(v > 0 for v in values)
    if npositive < 5:
        raise ValueError("At least five positive measured concentrations required")
    path = column.get("QuantityInfo", {}).get("Path", "")
    text = (observation.get("Name", "") + "|" + path).lower()
    reasons = []
    if props.get("Species") != "Human":
        reasons.append("Human observation species not explicitly documented")
    if "|plasma|" not in path.lower() or any(s in text for s in ("whole blood", "arterial plasma", "unbound", "urine")):
        reasons.append("Primary observation is not unambiguously total venous plasma")
    if str(props.get("Organ", "")).replace(" ", "").lower() not in {"peripheralvenousblood", "venousblood"}:
        reasons.append("Venous sampling site not explicitly documented")
    expected = "ocd" if expected_drug == "Oxycodone" else expected_drug.lower()
    # Older snapshots omit the redundant Molecule extended property while
    # still explicitly naming the analyte in their observation quantity path.
    analyte = props.get("Molecule") or (path.split("|")[-2] if "|" in path else "")
    aliases = {expected}
    if expected_drug == "Clarithromycin" and "|plasma|clarithromycin|" in path.lower():
        aliases.add("clarithromycin test old")  # Exact obsolete display alias in Chu source.
    if str(analyte).lower() not in aliases:
        reasons.append("Observed parent identity is not explicitly matched")
    return {
        "column_index": 0, "time_unit": grid["Unit"], "concentration_unit": column["Unit"],
        "concentration_dimension": column.get("Dimension"), "quantity_path": path,
        "sampling_time_h": [t if tu == "h" else t / 60 for t in times],
        "point_count": len(times), "positive_point_count": npositive,
        "matrix": "plasma" if not reasons else "requires_review", "species": props.get("Species"),
        "sample_size": props.get("N"), "statistic": props.get("Data type") or path.rsplit("|", 1)[-1],
        "primary_reference": props.get("Reference"), "study_id": props.get("Study Id"),
        "source_location": props.get("Source"), "blockers": reasons,
    }


def human_metadata(individual):
    origin = individual["OriginData"]
    if origin.get("Species") != "Human":
        raise ValueError("Only human individuals are eligible")
    sex = origin.get("Gender")
    if sex not in {"MALE", "FEMALE"}:
        raise ValueError("Human sex must be explicit")
    output = {"sex": sex, "population": origin.get("Population", "European_ICRP_2002"), "seed": 17}
    assumed, source_fields = [], {}
    for source, field, expected_unit, default in (("Age", "age_years", "year(s)", 30.),
            ("Weight", "weight_kg", "kg", 70.), ("Height", "height_cm", "cm", 175.)):
        v = origin.get(source)
        if v is None:
            output[field] = default
            assumed.append(field)
            source_fields[field] = {"basis": "predeclared_engineering_assumption", "value": default}
        else:
            if v.get("Unit") != expected_unit or not finite(v.get("Value")) or v["Value"] <= 0:
                raise ValueError("Unsupported human metadata: " + source)
            output[field] = v["Value"]
            source_fields[field] = {"basis": "snapshot_clinical_metadata_not_independently_verified",
                                    "value": v["Value"], "value_origin": v.get("ValueOrigin", origin.get("ValueOrigin"))}
    if output["age_years"] < 18:
        raise ValueError("Adult clinical cohort required")
    return output, {"individual_name": individual["Name"], "fields": source_fields,
                    "assumed_fields": assumed, "clinical_demographics_verified": False}


def dosing_metadata(protocol, human):
    if protocol.get("ApplicationType") not in {"Intravenous", "IntravenousBolus"} or protocol.get("DosingInterval") != "Single" or protocol.get("Schemas"):
        raise ValueError("Explicit single-dose IV protocol required")
    parameters = {p["Name"]: p for p in protocol.get("Parameters", [])}
    dose = parameters["InputDose"]
    if not finite(dose.get("Value")) or dose["Value"] <= 0 or unit(dose.get("Unit")) not in {"mg", "mg/kg"}:
        raise ValueError("Explicit positive mg or mg/kg dose required")
    start = parameters.get("Start time", {"Value": 0., "Unit": "h"})
    if start.get("Value") != 0 or unit(start.get("Unit")) != "h":
        raise ValueError("Single-dose clock must start at zero")
    infusion = 0.
    if protocol["ApplicationType"] == "Intravenous":
        x = parameters.get("Infusion time", {})
        if not finite(x.get("Value")) or x["Value"] <= 0 or unit(x.get("Unit")) not in {"min", "h"}:
            raise ValueError("Infusion duration must be explicit")
        infusion = x["Value"] * (60 if unit(x["Unit"]) == "h" else 1)
    value = dose["Value"] * (human["weight_kg"] if unit(dose["Unit"]) == "mg/kg" else 1)
    return {"route": "intravenous", "dose_mg": value, "start_h": 0., "infusion_minutes": infusion}, {
        "protocol_name": protocol["Name"], "source_dose_value": dose["Value"], "source_dose_unit": dose["Unit"],
        "source_application_type": protocol["ApplicationType"], "salt_to_parent_conversion_applied": False,
        "dose_basis": "source_protocol_parent_equivalent_requires_primary_review",
    }


def _history(root):
    paths = list((root / "experiments/blueprint_design/development_inputs").rglob("input_package.json"))
    paths += [root / "experiments/blueprint_design/development_inputs/multidrug_fixed_v1/input_manifest.json"]
    history = {name: [] for name in CHOICES}
    for path in paths:
        # Existing package membership only; no scores or prediction results.
        data = json.loads(path.read_text())
        cases = data.get("cases", [data.get("case", {})])
        for case in cases:
            requirement = case.get("requirement", case.get("design_request", {}).get("requirement", {}))
            name = case.get("compound_name", requirement.get("name"))
            if name in history:
                history[name].append(str(path.relative_to(root)))
    for relative in ("experiments/blueprint_design/source_context_pilot_20260921/protocol.json",
                     "experiments/blueprint_design/pk_guided_pilot_20260921/protocol.json"):
        path = root / relative
        if not path.exists():
            continue
        identifiers = json.loads(path.read_text()).get("cases", [])
        for name in history:
            if any(isinstance(case, str) and case.lower().startswith(name.lower() + "_") for case in identifiers):
                history[name].append(relative)
    return history


def curate(root=ROOT):
    root = Path(root).resolve(strict=True)
    inventory_path, old_path = inside(root, INVENTORY), inside(root, OLD_PROTOCOL)
    inventory = json.loads(inventory_path.read_text())
    old = json.loads(old_path.read_text())
    old_names = {c["drug"].lower() for c in old["cases"]}
    sources = {INVENTORY: sha(inventory_path), OLD_PROTOCOL: sha(old_path)}
    history = _history(root)
    screening, cases = [], []
    for drug, choice in CHOICES.items():
        alias = "OCD" if drug == "Oxycodone" else drug
        entries = [e for e in inventory["entries"] if e.get("compound_names") == [alias]
                   and any(s["observed_name"].startswith(choice) for s in e.get("eligible_scenarios", []))]
        if len(entries) != 1 or drug.lower() in old_names:
            raise ValueError("Candidate inventory or old-cohort exclusion changed: " + drug)
        entry = entries[0]
        source = inside(root, entry["path"])
        if sha(source) != entry["sha256"]:
            raise ValueError("Frozen observation source changed: " + str(source))
        sources[entry["path"]] = entry["sha256"]
        scenarios = [s for s in entry["eligible_scenarios"] if s["observed_name"].startswith(choice)]
        if len(scenarios) != 1:
            raise ValueError("Candidate clinical scenario is ambiguous: " + drug)
        scenario = scenarios[0]
        snapshot = json.loads(source.read_text())
        oi, observation = named(snapshot["ObservedData"], scenario["observed_name"])
        ii, individual = named(snapshot["Individuals"], scenario["individual"])
        pi, protocol = named(snapshot["Protocols"], scenario["protocol"])
        # Deliberately never access snapshot['Compounds'], ['Simulations'], etc.
        clinical = observation_metadata(observation, expected_drug=drug)
        human, human_audit = human_metadata(individual)
        dosing, dosing_audit = dosing_metadata(protocol, human)
        blockers = list(dict.fromkeys(clinical.pop("blockers") + PENDING.get(drug, [])))
        ref = {"path": entry["path"], "sha256": entry["sha256"]}
        locator = {"source": ref, "observation_index": oi, "observation_name": observation["Name"],
                   **{k: clinical[k] for k in ("column_index", "time_unit", "concentration_unit", "concentration_dimension", "quantity_path")}}
        study = clinical["study_id"] or STUDY_NAMES.get(drug, scenario.get("study_id")) or "unidentified clinical study"
        slug = re.sub(r"[^a-z0-9]+", "_", drug.lower()).strip("_")
        case = {"case_id": slug + "_expanded_iv", "drug": drug, "compound_id": None,
            "study_id": study, "cluster_id": study, "cohort": "expanded", "dataset": "osp_observed_expansion",
            "reference_kind": "observed_curve", "reference_locator": locator,
            "human": human, "human_provenance": human_audit, "dosing": dosing, "dosing_provenance": dosing_audit,
            "metadata_locators": {"source": ref, "individual_index": ii, "protocol_index": pi},
            "nca_policy": {"sampling_time_h": clinical["sampling_time_h"], "lloq_ng_ml": None,
                "sampling_schedule_kind": "digitized_clinical_times_not_verified_nominal_schedule",
                "integration": "paired observed window; linear-up/log-down for both curves; no synthetic predose point",
                "reported_original_scalar_reference_count": 0, "no_original_reference_recalculation": True},
            "target": {"window_start_h": 0., "window_end_h": max(clinical["sampling_time_h"])},
            "observation_metadata": {k:v for k,v in clinical.items() if k not in ("sampling_time_h", "source_location")},
            "primary_reference": clinical["primary_reference"] or PRIMARY_LINKS.get(drug),
            "primary_review_complete": False, "clinical_metadata_eligible": not blockers,
            "qualification_flags": ["historically_screened_snapshot_not_blinded", "no_original_scalar_PK_references",
                                    "independent_chemical_inputs_still_required", "primary_population_and_dose_review_pending"],
            "historical_development_used": bool(history[drug]), "historical_membership_sources": history[drug],
            "historical_membership_interpretation": "Positive findings only; no matching package does not establish a previously unseen compound",
            "blockers": blockers, "construction_sources": [], "ready_for_prediction": False}
        if human_audit["assumed_fields"]:
            case["qualification_flags"].append("missing_demographics_use_declared_engineering_defaults")
        if drug == "Ethinylestradiol":
            case["qualification_flags"].append("Back1981_possible_levonorgestrel_coadministration_requires_review")
        if drug == "Paroxetine" or clinical["statistic"].lower() == "individual":
            case["qualification_flags"].append("single_subject_not_population_mean_source_sensitivity")
        screening.append(case)
        for relative in history[drug]:
            sources[relative] = sha(root / relative)
        if not blockers:
            cases.append(case)
    return {"version": VERSION,
        "selection_rule": "One predeclared IV parent total-venous-plasma series per new compound; >=5 positive points; clinical metadata only, no scores",
        "minimum_sample_rule": "No lowering of qualification rules to reach a desired number of drugs",
        "historical_scope": "Local sources have been screened before; this is expanded development validation, not a new blinded study",
        "default_missing_demographics": {"age_years":30.,"weight_kg":70.,"height_cm":175.},
        "construction_boundary": "No original Compound parameters, fitted logP/fu, CL, Vss, half-life, model curves or model processes copied",
        "reference_endpoint_scope": ["plasma_concentration_time", "observed_window_auc", "sampled_cmax", "sampled_tmax", "last_observed_time_concentration"],
        "unavailable_reference_endpoints": ["terminal_half_life","auc_inf","clearance","vz","mrt","vss"],
        "original_drugs_excluded": sorted(old_names), "cases":cases, "screening":screening,
        "counts": {"screened":len(screening),"clinical_metadata_eligible":len(cases),"pending_or_excluded":len(screening)-len(cases),
                   "ready_for_prediction":0}, "source_pins":sources,
        "code_pins":{str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__))},
        "model_fitted":False,"backend_started":False,"target_values_exported":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default=OUTPUT)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    result=curate(ROOT)
    path=inside(ROOT,args.output)
    if not args.dry_run:
        if path.exists() and json.loads(path.read_text())!=result:
            raise ValueError("Existing cohort differs; do not silently replace a frozen selection")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"path":str(path.relative_to(ROOT)),**result["counts"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
