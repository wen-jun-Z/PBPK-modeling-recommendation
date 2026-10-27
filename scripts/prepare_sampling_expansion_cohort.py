#!/usr/bin/env python3
"""Curate thirty distinct IV drugs without copying source-model drug parameters.

The archive adapter reads only ObservedData, nominal Protocols, Individuals'
OriginData and simulation-to-building-block names. Published observations are
never called model predictions. New difficult populations/arms are retained as
declared sensitivity strata; thirty drugs is not thirty independent studies.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "experiments/blueprint_design/sampling_expansion_20261009/inputs"
INVENTORY = ROOT / "experiments/blueprint_design/development_inputs/cohort_screening_20260918/source_inventory.json"
EXISTING = ROOT / "experiments/blueprint_design/expanded_tail_20260930/inputs/unified_manifest.json"
ENDPOINTS = ["plasma_concentration_time", "auc", "cmax", "tmax", "c_last",
             "terminal_half_life", "auc_inf", "clearance", "vz", "mrt", "vss"]

# Chosen by bibliographic identifiers and nominal dose/arm, before new fits.
# These strings are exact names in the public OSP repository snapshots.
SPECS = {
    "Ethinylestradiol": {
        "development": ("Ethinylestradiol 0.05 mg iv", "Ethinylestradiol.Back1979.0.05.iv.Mean"),
        "target": ("Ethinylestradiol 0.03 mg iv", "Ethinylestradiol.Orme1991.0.03.iv.Mean"),
        "flags": ["repository_digitized_mean", "generic_female_reference"]},
    "Finerenone": {
        "development": ("F - 0.25 mg IV", "F A.Finerenone.Finerenone 0.25mg i.v..arith Mean"),
        "target": ("F - 1 mg IV", "F B.Finerenone.Finerenone 1mg i.v..arith Mean"),
        "flags": ["within_study_distinct_arms", "participant_overlap_unresolved", "repository_study_code_not_bibliographic_identifier"],
        "cluster_id": "Finerenone_trial_F", "cross_study": False},
    "Levonorgestrel": {
        "development": ("Back 1981_Levonorgestrel_IV_0.25 mg", "Back 1981_Levonorgestrel_IV_0.25 mg + 50 µg EE"),
        "target": ("Cicali 2021_Levonorgestrel_IV_0.09 mg", "Cicali 2021_Levonorgestrel_IV_0.09 mg"),
        "target_study": "Cicali2021", "flags": ["development_ethinylestradiol_coadministration", "SHBG_context_not_mechanistically_modelled", "generic_female_reference"],
        "development_metadata_support": "https://pubmed.ncbi.nlm.nih.gov/6786829/"},
    "Mexiletine": {
        "development": ("Mexiletine 200 mg iv 5 min", "Campbell1978-clinpharm.Mexiletine.200.iv.Mexiletine"),
        "target": ("Mexiletine 200 mg iv 30 min", "Pentikainen1984.Mexiletine.200HCL.iv.recovery myocardial infarction_gr2"),
        "flags": ["recovery_after_myocardial_infarction", "parent_equivalent_HCl_dose", "generic_reference_human"],
        "target_metadata_support": "https://pubmed.ncbi.nlm.nih.gov/6199590/"},
    "Moclobemide": {
        "development": ("50mg iv", "Raaflaub1984.Moclobemide.50.iv_inf_2min.sd.PatID.1"),
        "target": ("150mg iv 20min", "Schoerlin1987.Moclobemide.150.iv_inf_20min.sd.day1.RepSubject"),
        "flags": ["individual_to_representative_subject", "repository_bolus_corrected_to_observation_two_minute_infusion", "day_time_unit_converted"],
        "development_metadata_support": "https://pubmed.ncbi.nlm.nih.gov/6538424/"},
    "Paclitaxel": {
        "development": ("Huizing1993_135mg_m2_3h", "#102_Huizing-1993_Fig.4_135mg_m2_3h."),
        "target": ("Huizing1995_250mg_m2_3h", "#101_Huizing-1995_Fig.2_250mg_m2_3h"),
        "flags": ["oncology_population", "BSA_protocol_conversion", "sex_from_observation_overrides_repository_male_template", "linear_model_may_miss_nonlinear_PK"]},
    "Probenecid": {
        "development": ("Dayton 1963 - Subject A_1860mg_i.v._s.d._1", "Dayton 1963 - 1860 mg IV Subject A - Probenecid - IV - 1860 mg - Plasma - indiv."),
        "target": ("Emanuelsson 1987_500mg_i.v._s.d._14", "Emanuelsson 1987 - 500 mg iv - Probenecid - IV - 500 mg - Plasma - agg. (n=6)"),
        "flags": ["individual_to_group_mean", "male_to_female_conditions", "potential_dose_dependent_clearance"]},
    "Rifampicin": {
        "development": ("Rifampicin iv 300 mg (3 h)", "Nitti 1977 - 300 mg - Rifampicin - IV - 300 mg - Serum - agg. (n=2)"),
        "target": ("Rifampicin iv 300 mg (0.5 h)", "sanofi-aventis U.S. LLC. 2013 - 300 mg - Rifampicin - IV - 300 mg - Plasma - agg. (n=12)"),
        "flags": ["serum_development_plasma_target", "small_development_group_n2", "no_autoinduction_validation"]},
    "Rosuvastatin": {
        "development": ("iv  8.0 mg, Martin 2003c, 4 h, n=10", "Martin 2003c - IV 8 mg control - Rosuvastatin - IV - 8 mg - Plasma - agg. (n=10)"),
        "target": ("iv  0.00174 mg, Billington 2019, bolus, n=1", "Billington 2019 - IV bolus 0.00174 mg control - Rosuvastatin - IV - 1.74E-3 mg - Arterial Plasma - indiv."),
        "flags": ["therapeutic_to_microdose", "peripheral_to_arterial_plasma", "group_mean_to_individual", "generic_reference_human"]},
    "Montelukast": {
        "archive": "artifacts/pbpk-model-assets/public_collection_20260909/Montelukast-Model__b65eff5296da/source/Montelukast.json",
        "development": ("Cheng 1996 3mg IV", "Cheng 1996 - 3mg IV male - Montelukast - IV - 3 mg - Plasma - agg. (n=9)"),
        "target": ("Cheng 1996 18mg IV", "Cheng 1996 - 18mg IV male - Montelukast - iv - 18 mg - Plasma - agg. (n=9)"),
        "development_study": "Cheng1996_male_3mg_arm", "target_study": "Cheng1996_male_18mg_arm",
        "flags": ["within_study_distinct_dose_arms", "participant_overlap_possible", "repository_digitized_mean"],
        "cluster_id": "Cheng1996", "cross_study": False},
}


def read(path):
    return json.loads(Path(path).read_text())


def safe(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Project-only path required")
    return p


def ref(path, location="$", **extra):
    p = safe(path)
    return {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "location": location, **extra}


def checked(source):
    p = safe(source["path"])
    if ref(p)["sha256"] != source["sha256"]:
        raise ValueError("Source hash mismatch: " + str(p))
    return read(p)


def save(path, doc):
    p = safe(path)
    if not p.is_relative_to(OUT):
        raise ValueError("Curation writes restricted to this batch inputs")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if p.exists() and p.read_text() != text and p.name != "cohort_manifest.json":
        raise ValueError("Refusing to replace changed curation artifact: " + str(p))
    if not p.exists() or p.read_text() != text:
        p.write_text(text)
    return ref(p)


def validate_rows(rows, minimum=5):
    out = sorted(rows, key=lambda r: r["time_min"])
    if len({r["time_min"] for r in out}) != len(out):
        raise ValueError("Duplicate clinical times need explicit aggregation")
    if any(not math.isfinite(float(r[k])) or float(r[k]) < 0
           for r in out for k in ("time_min", "concentration_umol_per_l")):
        raise ValueError("Invalid clinical time/concentration")
    if sum(r["time_min"] > 0 and r["concentration_umol_per_l"] > 0 for r in out) < minimum:
        raise ValueError("Insufficient unique positive observed times")
    return out


def archive_curve(drug, phase, identity, inventory):
    spec = SPECS[drug]
    entries = [e for e in inventory["entries"] if e["compound_names"] == [drug]]
    if spec.get("archive"):
        entries = [e for e in entries if e["path"] == spec["archive"]]
    if len(entries) != 1:
        raise ValueError("Ambiguous one-compound archive: " + drug)
    source = entries[0]
    archive = checked(source)
    sim_name, obs_name = spec[phase]
    simulations = [s for s in archive["Simulations"] if s["Name"] == sim_name]
    if len(simulations) != 1:
        raise ValueError("Unknown nominal simulation: " + sim_name)
    simulation = simulations[0]
    if obs_name not in simulation.get("ObservedData", []):
        raise ValueError("Observation is not linked to nominal protocol")
    observation = next(o for o in archive["ObservedData"] if o["Name"] == obs_name)
    meta = {z["Name"]: z.get("Value") for z in observation.get("ExtendedProperties", [])}
    species, matrix = meta.get("Species"), meta.get("Compartment")
    assumptions = []
    if drug == "Levonorgestrel" and phase == "development":
        if species is not None or matrix is not None:
            raise ValueError("Reviewed levonorgestrel metadata changed")
        species, matrix = "Human", "Plasma"
        assumptions.append("Human women and plasma matrix verified in Back1981 PMID6786829; absent archive fields are repaired explicitly.")
    if species != "Human" or matrix != "Plasma":
        raise ValueError("Human plasma identity unresolved")
    columns = [c for c in observation["Columns"] if c.get("DataInfo", {}).get("Origin") == "Observation"]
    if len(columns) != 1:
        raise ValueError("Exactly one observation column required")
    column, grid = columns[0], observation["BaseGrid"]
    mw = float(identity["molecular_weight_g_mol"])
    factors = {"mg/l": 1000 / mw, "µg/ml": 1000 / mw, "ug/ml": 1000 / mw,
               "µg/l": 1 / mw, "ug/l": 1 / mw, "ng/ml": 1 / mw,
               "µmol/l": 1., "umol/l": 1.}
    times = {"h": 60., "min": 1., "s": 1/60., "day(s)": 1440.}
    if column["Unit"].lower() not in factors or grid["Unit"] not in times:
        raise ValueError("Unreviewed observation unit")
    if len(grid["Values"]) != len(column["Values"]):
        raise ValueError("Concentration/time length mismatch")
    tf, cf = times[grid["Unit"]], factors[column["Unit"].lower()]
    rows = [{"time_min": float(t)*tf, "concentration_umol_per_l": float(c)*cf}
            for t, c in zip(grid["Values"], column["Values"])]
    individual = next(i for i in archive["Individuals"] if i["Name"] == simulation["Individual"])
    origin = individual["OriginData"]
    if origin.get("Species") != "Human":
        raise ValueError("Nominal individual must be human")
    human = {"age_years": float(origin.get("Age", {}).get("Value", 30)),
             "weight_kg": float(origin.get("Weight", {}).get("Value", 70)),
             "height_cm": float(origin.get("Height", {}).get("Value", 175)),
             "sex": origin.get("Gender", "MALE"),
             "population": "European_ICRP_2002", "seed": 17}
    if drug == "Paclitaxel":
        observed_sex = str(meta.get("Gender", "")).upper()
        if observed_sex not in {"MALE", "FEMALE"}:
            raise ValueError("Paclitaxel clinical sex unresolved")
        human["sex"] = observed_sex
        assumptions.append("Clinical observation sex overrides generic male source-model individual; reference weight/height remain explicit assumptions.")
    compound_links = simulation["Compounds"]
    if len(compound_links) != 1:
        raise ValueError("Only one parent-drug protocol permitted")
    protocol_name = compound_links[0]["Protocol"]["Name"]
    protocol = next(p for p in archive["Protocols"] if p["Name"] == protocol_name)
    if protocol["ApplicationType"] not in {"Intravenous", "IntravenousBolus"}:
        raise ValueError("Only IV protocol accepted")
    pars = {p["Name"]: p for p in protocol["Parameters"]}
    if pars["Start time"]["Value"] != 0:
        raise ValueError("Dosing must begin at zero")
    dm = {"mg": 1., "g": 1000., "µg": .001, "mg/kg": human["weight_kg"],
          "µg/kg": .001*human["weight_kg"]}
    if pars["InputDose"]["Unit"] not in dm:
        raise ValueError("Unreviewed dose unit")
    dose = float(pars["InputDose"]["Value"])*dm[pars["InputDose"]["Unit"]]
    infusion = pars.get("Infusion time", {"Value": 0., "Unit": "min"})
    if infusion["Unit"] not in times:
        raise ValueError("Unreviewed infusion unit")
    infusion_min = float(infusion["Value"])*times[infusion["Unit"]]
    if drug == "Moclobemide" and phase == "development":
        if "iv_inf_2min" not in str(meta.get("Route")):
            raise ValueError("Explicit two-minute observation route required")
        infusion_min = 2.
        assumptions.append("Observation route iv_inf_2min overrides generic bolus protocol; no fitted drug parameter is copied.")
    if protocol.get("DosingInterval") != "Single":
        until = spec.get(phase + "_until_h")
        if drug != "Vancomycin" or phase != "development" or until != 12. or protocol.get("DosingInterval") != "DI_12_12":
            raise ValueError("Unreviewed repeated-dose protocol")
        rows = [r for r in rows if r["time_min"] < until*60]
        assumptions.append("Only measurements before second dose at12h are used; repeated-dose observations are excluded, not simulated as single dose.")
    if drug == "Mexiletine" and phase == "target":
        if not math.isclose(dose, 166.2):
            raise ValueError("Reviewed mexiletine parent-equivalent dose changed")
        # PubMed6199590 explicitly reports 200mg hydrochloride. Archive's
        # nominal166.2mg parent corresponds to the independently known salt.
        assumptions.append("Pentikainen1984 reports200mg mexiletine HCl; archive nominal166.2mg parent equivalent retained, with recovery-AMI population flag.")
    source_matrix = matrix.lower()
    if drug == "Rifampicin" and phase == "development":
        source_matrix = "serum"
        assumptions.append("Source observation is serum despite archive Compartment=Plasma; serum-to-plasma equality is an explicit sensitivity assumption, not a measured fact.")
    if "Arterial Plasma" in obs_name:
        assumptions.append("Arterial plasma target is retained as matrix/site sensitivity; generic plasma predictions do not establish arterial/peripheral equivalence.")
    rows = validate_rows(rows)
    study = spec.get(phase + "_study", meta.get("Study Id"))
    if not study:
        raise ValueError("Explicit clinical study identifier required")
    dosing = {"route": "intravenous", "dose_mg": dose, "start_h": 0., "infusion_minutes": infusion_min}
    common = ["Public repository digitized clinical concentrations; not model-generated concentrations.",
              "Only observations, nominal dosing and OriginData demographics exported; no source fitted drug parameters/processes copied.",
              "Absent mean demographics use declared reference-adult assumptions; no claim to recover participant covariates."]
    return {"compound_id": "pubchem:" + str(identity["pubchem_cid"]), "study_id": study,
            "role": "calibration_data_reconstruction" if phase == "development" else "historical_cross_study_reference",
            "matrix": matrix.lower(), "source_matrix": source_matrix,
            "matrix_equivalence_assumed": source_matrix != matrix.lower(),
            "time_unit": "min", "unit": "umol/l",
            "molecular_weight_g_mol": mw, "human": human, "dosing": dosing,
            "observations": rows, "source": ref(source["path"], "ObservedData[" + obs_name + "]"),
            "metadata_source": ref(source["path"], "Protocols[" + protocol_name + "];Individuals.OriginData; named observation metadata"),
            "clinical_reference_url": spec.get(phase + "_metadata_support", meta.get("Reference")),
            "conversion": {"source_time_unit": grid["Unit"], "source_concentration_unit": column["Unit"],
                           "time_multiplier": tf, "concentration_multiplier": cf, "parent_molecular_weight_g_mol": mw},
            "source_observation_name": obs_name, "source_simulation_name": sim_name,
            "limitations": common + assumptions + spec["flags"],
            "not_an_original_reported_scalar_reference": True}


def prepare():
    original, inventory = read(EXISTING), read(INVENTORY)
    cases = deepcopy(original["cases"])
    audit = []
    for case in cases:
        case["expansion_cohort"] = "previous_ten" if case.get("cohort") == "original" else "previous_twenty_extension"
        case["independent_of_prior_algorithm_development"] = False
        dev = checked(case["sampling_sources"]["full"])
        validate_rows(dev["observations"])
        audit.append({"drug": case["drug"], "status": "existing_curated_pair", "case_id": case["case_id"]})
    for drug, spec in SPECS.items():
        phys_path = OUT / "physchem" / (drug.lower() + ".json")
        if not phys_path.exists():
            audit.append({"drug": drug, "status": "pending_independent_physchem", "expected_path": str(phys_path.relative_to(ROOT))})
            continue
        try:
            physical = read(phys_path)
            identity = physical["identity"]
            dev = archive_curve(drug, "development", identity, inventory)
            target = archive_curve(drug, "target", identity, inventory)
            if dev["study_id"] == target["study_id"]:
                raise ValueError("Identical source study/arm labels")
            folder = OUT / "cohort" / drug.lower()
            dref, tref = save(folder / "development.json", dev), save(folder / "reference.json", target)
            times = [r["time_min"]/60. for r in target["observations"]]
            cid = drug.lower() + "_sampling_expansion_iv"
            case = {"case_id": cid, "compound_id": dev["compound_id"], "drug": drug,
                    "cohort": "extension", "expansion_cohort": "new_ten_source_curated",
                    "study_id": target["study_id"], "development_study_id": dev["study_id"],
                    "cluster_id": spec.get("cluster_id", target["study_id"]),
                    "cross_study_pair": spec.get("cross_study", True),
                    "physchem": str(phys_path.relative_to(ROOT)), "physchem_source": ref(phys_path),
                    "development": dref["path"], "reference": tref["path"],
                    "development_source": dref, "reference_source": tref,
                    "sampling_sources": {"full": dref}, "reference_kind": "observed_curve",
                    "target_request_metadata": {"human": target["human"], "dosing": target["dosing"],
                        "window_end_h": max(24., math.ceil(times[-1])), "demographic_assumptions": target["limitations"]},
                    "nca_policy": {"sampling_time_h": times, "sampling_schedule_kind": "source_digitized_observed_times_not_verified_nominal_schedule",
                        "dose_ng": target["dosing"]["dose_mg"]*1e6, "infusion_h": target["dosing"]["infusion_minutes"]/60.,
                        "weight_kg": target["human"]["weight_kg"], "lloq_ng_ml": None,
                        "source": tref, "integration": "linear_up_log_down",
                        "reported_original_scalar_reference_count": 0,
                        "no_invented_zero_concentration": True},
                    "sensitivity_flags": spec["flags"], "approximations": target["limitations"],
                    "renal_enabled": False, "requests_pending_RF_priors": True,
                    "source_model_fitted_parameters_copied": False,
                    "independent_of_prior_algorithm_development": False,
                    "newly_downloaded_clinical_observations": False}
            cases.append(case)
            audit.append({"drug": drug, "status": "clinical_pair_prepared", "case_id": cid,
                          "development_positive_points": sum(r["time_min"] > 0 and r["concentration_umol_per_l"] > 0 for r in dev["observations"]),
                          "target_positive_points": sum(r["time_min"] > 0 and r["concentration_umol_per_l"] > 0 for r in target["observations"]),
                          "development_study": dev["study_id"], "target_study": target["study_id"],
                          "cross_study_pair": case["cross_study_pair"], "flags": spec["flags"]})
        except (ValueError, KeyError, StopIteration) as exc:
            audit.append({"drug": drug, "status": "curation_blocked", "reason": str(exc) or type(exc).__name__})
    audit.append({"drug": "Propofol", "status": "excluded_before_fitting",
                  "reason": "Gepts1987 whole-blood name conflicts with Plasma metadata; Schnider1998 has mixed bolus+infusion. No unverified blood/plasma conversion or dosing simplification."})
    audit.append({"drug": "Vancomycin", "status": "excluded_before_fitting",
                  "reason": "Six independently documented ionization sites exceed frozen native input maximum three; no site truncation or neutralization to force enrollment. A clinical pair exists but construction is unsupported."})
    result = {"version": "sampling-expansion-cohort-20261009-1", "target_distinct_drugs": 30,
              "unique_drugs": len({c["compound_id"] for c in cases}), "cases": cases,
              "candidate_audit": audit, "source_pins": {ref(p)["path"]: ref(p)["sha256"] for p in (EXISTING, INVENTORY, Path(__file__))},
              "original_reported_and_observation_derived_endpoints_separate": True,
              "same_study_arms_and_matrix_population_sensitivity_separate": True,
              "algorithm_outcomes_used_to_select_cases": False, "native_fits_executed": False,
              "interpretation": "Expanded historical development evaluation; independent drug count is not an independent-study count. Original clinical scalar labels are not asserted from archive curves."}
    # Allow resume after requests have been attached without losing them.
    existing = OUT / "cohort_manifest.json"
    if existing.exists():
        prior_doc = read(existing)
        prior = {c["case_id"]: c for c in prior_doc.get("cases", [])}
        for c in cases:
            old = prior.get(c["case_id"], {})
            if c.get("development_source") == old.get("development_source"):
                for key in ("requests", "requests_pending_RF_priors", "construction_preflight_passed", "RF_exclusion_source"):
                    if key in old and (key not in c or key in {"requests", "requests_pending_RF_priors"}):
                        c[key] = old[key]
        if len(cases) == prior_doc.get("requests_ready") and all(c.get("construction_preflight_passed") for c in cases):
            for key in ("requests_ready", "RF_priors"):
                if key in prior_doc:
                    result[key] = prior_doc[key]
    save(existing, result)
    return result


def create_requests():
    """Attach one uniform 30-drug-excluded RF prior to each clinical pair.

    Measured/independent construction evidence remains fixed. Only an existing
    predicted CL/fu, or a missing new-drug CL/fu, is supplied by this new RF.
    """
    from pbpk_kb.blueprints.evidence_blueprint import assemble_evidence_blueprint
    manifest = prepare()
    if manifest["unique_drugs"] != 30:
        raise ValueError("All thirty clinical pairs must be curated before request freezing")
    prior_path = OUT / "priors/predictions.json"
    prior = read(prior_path)
    estimates = prior["external_predictions"]
    missing = {c["drug"] for c in manifest["cases"]} - set(estimates)
    if missing:
        raise ValueError("Missing target-excluded RF estimates: " + repr(sorted(missing)))
    source = ref(prior_path, "external_predictions; all thirty identity/scaffold groups excluded")
    method = {"name": "random_forest", "version": "sampling-expansion30-20261009",
              "training_overlap": False,
              "applicability": {"target_structure_and_scaffold_excluded": True,
                                "human_IV_parameter_model": True,
                                "clinical_curves_used_for_training": False}}
    assumptions = save(OUT / "construction_assumptions.json", {
        "solubility_mg_ml": 1., "solubility_ph": 7., "binding_partner": "Unknown",
        "elimination_scope": "hepatic_only",
        "scope": "Engineering IV-only input placeholders and lumped total clearance allocation; not measured drug parameters."})
    for case in manifest["cases"]:
        drug, cid = case["drug"], case["compound_id"]
        pred = estimates[drug]
        if pred.get("training_structure_overlap") is not False or pred.get("training_scaffold_overlap") is not False:
            raise ValueError("RF structure/scaffold exclusion not verified: " + drug)
        values = pred["predictions"]
        folder = OUT / "cohort" / drug.lower()
        is_new = case["expansion_cohort"] == "new_ten_source_curated"
        if is_new:
            physical = read(ROOT / case["physchem"])
            if physical.get("construction_status") not in {None, "ready", "supported", "construction_ready"}:
                raise ValueError("Chemical construction blocked: " + drug + " " + str(physical.get("construction_status")))
            evidence = deepcopy(physical["evidence"])
            for item in evidence:
                item["compound_id"] = cid
            present = {e["parameter"] for e in evidence}
            for parameter, value, unit in (("solubility", 1., "mg/ml"), ("solubility_ph", 7., "1"),
                                            ("binding_partner", "Unknown", "text"), ("elimination_scope", "hepatic_only", "text")):
                if parameter not in present:
                    evidence.append({"parameter": parameter, "compound_id": cid, "value": value,
                                     "unit": unit, "basis": "assumption", "source": assumptions,
                                     "notes": "IV-only engineering placeholder, not an observed clinical value."})
            dev = checked(case["development_source"])
            target = checked(case["reference_source"])
            template = {"entry_reason": "model_missing", "compound_id": cid, "compound_name": drug,
                        "human": target["human"], "dosing": target["dosing"], "partition_method": "PKSimStandard",
                        "target": {"endpoints": ENDPOINTS, "primary_endpoint": "plasma_concentration_time",
                                   "window_start_h": 0., "window_end_h": min(168., max(24., math.ceil(target["observations"][-1]["time_min"]/60.)))},
                        "forbidden_sources": [case["reference_source"]],
                        "assumptions": ["Historical drug/condition evaluation, not a new blind clinical test.",
                            "No source PBPK fitted parameters copied; independent chemical evidence and structure-only RF priors.",
                            "All30 compound structures and scaffold groups excluded from RF training.",
                            "Visible-only concentration calibration; no full-development NCA summary prior.",
                            "No organ endpoint, disease mechanism or clinical decision reliability established.",
                            *case["sensitivity_flags"]]}
            development_template = deepcopy(template)
            development_template["human"], development_template["dosing"] = dev["human"], dev["dosing"]
            development_template["target"]["window_end_h"] = min(168., max(template["target"]["window_end_h"], math.ceil(dev["observations"][-1]["time_min"]/60.)))
        else:
            # Always originate from the previous frozen manifest, not this
            # mutable curation receipt, so repeated prepare is deterministic.
            legacy = next(c for c in read(EXISTING)["cases"] if c["case_id"] == case["case_id"])
            template = checked(legacy["requests"]["evidence"]["target"])
            development_template = checked(legacy["requests"]["evidence"]["development"])
            evidence = deepcopy(template["evidence"])
            if evidence != development_template["evidence"]:
                raise ValueError("Development/target construction evidence differs before calibration: " + drug)
        for parameter, endpoint, unit in (("total_clearance", "human_CL_mL_min_kg", "ml/min/kg"),
                                          ("fraction_unbound", "human_fup", "fraction")):
            old = [e for e in evidence if e["parameter"] == parameter]
            if len(old) > 1:
                raise ValueError("Duplicate construction parameter")
            if old and old[0]["basis"] != "predicted":
                continue
            value = float(values[endpoint]["random_forest"])
            if not math.isfinite(value) or value <= 0 or (parameter == "fraction_unbound" and value > 1):
                raise ValueError("Invalid RF prediction: " + parameter)
            evidence = [e for e in evidence if e["parameter"] != parameter]
            evidence.append({"parameter": parameter, "compound_id": cid, "value": value, "unit": unit,
                             "basis": "predicted", "source": source, "prediction_method": method,
                             "context": {"species": "human", "matrix": "plasma", "route": "intravenous"},
                             "notes": "All30 structures/scaffolds excluded; no same-drug development or target concentration used for RF."})
        ledger = save(folder / "evidence.json", {"evidence": evidence})
        requests = {}
        for phase, req in (("development", development_template), ("target", template)):
            req["evidence"], req["evidence_ledger"] = deepcopy(evidence), ledger
            # Source refs only, no optional curator-only annotations.
            req["forbidden_sources"] = [{k: s[k] for k in ("path", "sha256", "location")} for s in req.get("forbidden_sources", [])]
            assembled = assemble_evidence_blueprint(req, ROOT)
            if assembled["status"] != "assembled":
                raise ValueError("Request preflight failed for " + drug + " " + phase + ": " + json.dumps(assembled, ensure_ascii=False)[:1000])
            requests[phase] = save(folder / (phase + "_request.json"), req)
        case["requests"] = {"evidence": requests}
        case["requests_pending_RF_priors"] = False
        case["construction_preflight_passed"] = True
        case["RF_exclusion_source"] = source
    manifest["RF_priors"] = source
    manifest["requests_ready"] = len(manifest["cases"])
    manifest["source_pins"][source["path"]] = source["sha256"]
    save(OUT / "cohort_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "requests", "status"])
    args = parser.parse_args()
    doc = prepare() if args.command == "prepare" else create_requests() if args.command == "requests" else read(OUT / "cohort_manifest.json")
    print(json.dumps({"unique_drugs": doc["unique_drugs"], "candidate_audit": doc["candidate_audit"]}, ensure_ascii=False, indent=2))
