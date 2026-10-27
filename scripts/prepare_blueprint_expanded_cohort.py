#!/usr/bin/env python3
"""Audit historical human IV pairs and prepare leakage-safe 20-drug RF priors.

This is curation, not a new blind clinical benchmark. No simulation is executed.
Only observed-data branches and nominal protocols are extracted from the OSP
archive; fitted compound parameters and expression profiles are never copied.
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
BASE = ROOT / "experiments/blueprint_design/optimization_20260929"
OLD = ROOT / "experiments/blueprint_design"
SOURCE_INVENTORY = OLD / "development_inputs/cohort_screening_20260918/source_inventory.json"
CURRENT = OLD / "evidence_policy_20260929/protocol.json"
WHEEL = ROOT / "artifacts/pk_predictors/pksmart/pksmart-3.0.1-py3-none-any.whl"
NAMES = ["caffeine", "alfentanil", "alprazolam", "fluconazole", "furosemide",
         "sufentanil", "digoxin", "sildenafil", "triazolam", "isoniazid"]
ENDPOINTS = ["plasma_concentration_time", "auc", "cmax", "tmax", "c_last",
             "terminal_half_life", "auc_inf", "clearance", "vz", "mrt", "vss"]

# These are pre-existing curation inputs, not outputs from successful fits.
LEGACY = {
    "caffeine": ("evidence_assembly_20260922/inputs/case_request.json",
        "multidrug_endpoints_20260923/inputs/caffeine/blanchard_observations.json",
        "evidence_assembly_20260922/inputs/wahllander_request.json",
        "evidence_assembly_20260922/inputs/wahllander_observations.json"),
    "alfentanil": ("multidrug_endpoints_20260923/inputs/alfentanil/phimmasone_2001_request.json",
        "multidrug_endpoints_20260923/inputs/alfentanil/phimmasone_2001_observations.json",
        "multidrug_endpoints_20260923/inputs/alfentanil/kharasch_1997_request.json",
        "multidrug_endpoints_20260923/inputs/alfentanil/kharasch_1997_observations.json"),
    "alprazolam": ("ten_drug_endpoints_20260923/inputs/alprazolam/alprazolam_kroboth1988_iv_0_5mg_request.json",
        "ten_drug_endpoints_20260923/inputs/alprazolam/alprazolam_kroboth1988_iv_0_5mg_observations.json", None, None),
    "fluconazole": ("multidrug_endpoints_20260923/inputs/fluconazole/fluconazole_shiba_1990_25mg_request.json",
        "multidrug_endpoints_20260923/inputs/fluconazole/fluconazole_shiba_1990_25mg_observations.json",
        "multidrug_endpoints_20260923/inputs/fluconazole/fluconazole_yeates_1994_healthy_request.json",
        "multidrug_endpoints_20260923/inputs/fluconazole/fluconazole_yeates_1994_healthy_observations.json"),
    "furosemide": ("ten_drug_endpoints_20260923/inputs/furosemide/furosemide_haegeli_2007_20mg_request.json",
        "ten_drug_endpoints_20260923/inputs/furosemide/furosemide_haegeli_2007_20mg_observations.json",
        "ten_drug_endpoints_20260923/inputs/furosemide/furosemide_waller_1982_40mg_request.json",
        "ten_drug_endpoints_20260923/inputs/furosemide/furosemide_waller_1982_40mg_observations.json"),
    "sufentanil": ("ten_drug_endpoints_20260923/inputs/sufentanil/sufentanil_bovill_1984_request.json",
        "ten_drug_endpoints_20260923/inputs/sufentanil/sufentanil_bovill_1984_observations.json",
        "ten_drug_endpoints_20260923/inputs/sufentanil/sufentanil_taverne_1992_request.json",
        "ten_drug_endpoints_20260923/inputs/sufentanil/sufentanil_taverne_1992_observations.json"),
}
RAW = {
    "furosemide": (("Hammarlund 1984", "i.v. 40 mg (bolus)_Hammarlund 1984_39+40"), None),
    "alprazolam": (None, ("Fleishaker 1989", "Fleishaker1989_IV_1mg")),
    "digoxin": (("Greiner 1999", "Digoxin iv, 1.0 mg, 0.5 h"),
                ("Wagner 1981", "Digoxin iv, 0.5 mg, 1 h")),
    "sildenafil": (("Nichols 2002", "IV 50mg"),
                   ("Muirhead 2002b", "IV 25mg", "25mg-IV-Infusion-Subject 1")),
    "triazolam": (("Kroboth 1995", "Kroboth1995_0.25mg_iv"),
                  ("Smith 1987", "Smith1987_0.25mg")),
    "isoniazid": (("Boxenbaum 1974", "Human_Boxenbaum 1974_subject A_9.87 mg/kg_IV_Slow"),
                  ("von Sassen 1985", "Human_von Sassen 1985_mean_10 mg/kg_IV_Slow")),
}


def read(path):
    return json.loads(Path(path).read_text())


def inside(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Resources must remain in the project")
    return p


def sha(path):
    return hashlib.sha256(inside(path).read_bytes()).hexdigest()


def ref(path, location="$", **extra):
    p = inside(path)
    return {"path": str(p.relative_to(ROOT)), "sha256": sha(p), "location": location, **extra}


def save(path, obj):
    p = inside(path)
    if not p.is_relative_to(BASE):
        raise ValueError("Output is restricted to this new batch")
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if p.exists() and p.read_text() != data:
        # The inventory is a preparation status, not a frozen simulation input.
        if p.name != "cohort_inventory.json":
            raise ValueError("Refusing to overwrite changed prepared artifact: " + str(p))
    p.write_text(data)
    return ref(p)


def evidence_sources(request):
    sources = [request["evidence_ledger"], *[e["source"] for e in request["evidence"]]]
    for s in sources:
        if sha(s["path"]) != s["sha256"]:
            raise ValueError("Existing source changed: " + s["path"])
    return sources


def identity(name):
    path = BASE / "sources" / (name + "_properties.json")
    item = read(path)["response"]["PropertyTable"]["Properties"][0]
    smiles = item.get("SMILES", item.get("IsomericSMILES"))
    if not smiles:
        raise ValueError("Molecular structure unavailable: " + name)
    return {"pubchem_cid": item["CID"], "molecular_weight_g_mol": float(item["MolecularWeight"]),
            "molecular_formula": item["MolecularFormula"], "isomeric_smiles": smiles,
            "xlogp": float(item["XLogP"]), "source": ref(path)}


def clean_observations(rows):
    values = []
    for row in rows:
        t, c = float(row["time_min"]), float(row["concentration_umol_per_l"])
        if not math.isfinite(t) or not math.isfinite(c) or t < 0 or c < 0:
            raise ValueError("Invalid clinical observation")
        values.append({"time_min": t, "concentration_umol_per_l": c})
    if len({x["time_min"] for x in values}) != len(values):
        raise ValueError("Duplicate clinical times require an explicit aggregation rule")
    values.sort(key=lambda x: x["time_min"])
    if sum(r["concentration_umol_per_l"] > 0 for r in values) < 5:
        raise ValueError("At least five positive observations required")
    return values


def legacy_curve(drug, req_path, obs_path, role):
    request, observations = read(OLD / req_path), read(OLD / obs_path)
    evidence_sources(request)
    if observations.get("matrix", "plasma") != "plasma":
        raise ValueError("Only total plasma can use this adapter")
    if observations.get("curve_evaluation_eligible") is False:
        raise ValueError("A curve rejected by existing source review cannot be used for fitting: " + obs_path)
    study = observations.get("study_id", observations.get("study_group_id"))
    if drug == "caffeine" and "blanchard" in obs_path:
        study = "Blanchard 1983"
    if not study:
        raise ValueError("Missing explicit study identifier")
    rows = clean_observations(observations["observations"])
    return {"compound_id": request["compound_id"], "study_id": study, "role": role,
        "matrix": "plasma", "time_unit": "min", "unit": "umol/l",
        "molecular_weight_g_mol": next(e["value"] for e in request["evidence"] if e["parameter"] == "molecular_weight"),
        "human": request["human"], "dosing": request["dosing"], "observations": rows,
        "source": ref(OLD / obs_path, "observations"),
        "metadata_source": ref(OLD / req_path, "human,dosing; no fitted parameter values copied"),
        "population": observations.get("population", "reference_adult_with_source_limitations"),
        "limitations": ["Historical developer-viewed clinical data; not a new blind test.",
            "Previously curated representative demographics and source-specific assumptions are retained.",
            *request.get("assumptions", [])]}, request


def raw_curve(drug, selector, role, inventory):
    record = next(x for x in inventory["entries"] if x["compound_names"] == [drug.capitalize()])
    if sha(record["path"]) != record["sha256"]:
        raise ValueError("Raw archive hash changed")
    candidates = [x for x in record["eligible_scenarios"]
                  if (x["study_id"], x["simulation"]) == selector[:2]
                  and (len(selector) == 2 or selector[2] in x["observed_name"])]
    if len(candidates) != 1:
        raise ValueError("Ambiguous study/arm choice: " + drug + repr(selector))
    pick, archive = candidates[0], read(ROOT / record["path"])
    observed = next(o for o in archive["ObservedData"] if o["Name"] == pick["observed_name"])
    meta = {e["Name"]: e.get("Value") for e in observed.get("ExtendedProperties", [])}
    if meta.get("Species", "").lower() != "human" or meta.get("Compartment", "").lower() != "plasma":
        raise ValueError("Human plasma metadata required")
    columns = [c for c in observed["Columns"] if c.get("DataInfo", {}).get("Origin") == "Observation"]
    if len(columns) != 1:
        raise ValueError("One declared observed concentration column required")
    column, grid = columns[0], observed["BaseGrid"]
    mw = identity(drug)["molecular_weight_g_mol"]
    # Observation values in OSP's JSON are expressed in the declared display
    # unit. Convert mass using verified parent molecular weight, never the
    # molecular weight in the fitted compound branch.
    factors = {"mg/l": 1000 / mw, "µg/ml": 1000 / mw, "ug/ml": 1000 / mw,
               "µg/l": 1 / mw, "ug/l": 1 / mw,
               "ng/ml": 1 / mw, "µmol/l": 1., "umol/l": 1.}
    if column["Unit"].lower() not in factors or grid["Unit"] not in {"h", "min"}:
        raise ValueError("Unreviewed concentration or time unit")
    factor, tf = factors[column["Unit"].lower()], 60. if grid["Unit"] == "h" else 1.
    if len(grid["Values"]) != len(column["Values"]):
        raise ValueError("Time/value length mismatch")
    rows = clean_observations([{"time_min": float(t) * tf,
        "concentration_umol_per_l": float(c) * factor} for t, c in zip(grid["Values"], column["Values"])])
    individual = next(i for i in archive["Individuals"] if i["Name"] == pick["individual"])
    origin = individual["OriginData"]
    human = {"age_years": float(origin.get("Age", {}).get("Value", 30)),
        "weight_kg": float(origin.get("Weight", {}).get("Value", 70)),
        "height_cm": float(origin.get("Height", {}).get("Value", 175)),
        "sex": origin.get("Gender", "MALE"), "population": "European_ICRP_2002", "seed": 17}
    protocol = next(p for p in archive["Protocols"] if p["Name"] == pick["protocol"])
    pars = {p["Name"]: p for p in protocol["Parameters"]}
    if protocol["ApplicationType"] not in {"Intravenous", "IntravenousBolus"} or protocol["DosingInterval"] != "Single":
        raise ValueError("Only a single IV dose is supported")
    dose, unit = float(pars["InputDose"]["Value"]), pars["InputDose"]["Unit"]
    dm = {"mg": 1., "µg": .001, "mg/kg": human["weight_kg"], "µg/kg": .001 * human["weight_kg"]}
    if unit not in dm:
        raise ValueError("Unreviewed dose unit")
    infusion = pars.get("Infusion time", {"Value": 0., "Unit": "min"})
    if infusion["Unit"] not in {"min", "h"}:
        raise ValueError("Unreviewed infusion unit")
    if pars["Start time"]["Value"] != 0:
        raise ValueError("Only first-dose zero-time alignment accepted")
    dosing = {"route": "intravenous", "dose_mg": dose * dm[unit], "start_h": 0.,
              "infusion_minutes": float(infusion["Value"]) * (60 if infusion["Unit"] == "h" else 1)}
    warnings = ["Historical OSP-repository digitized human observations, not new blind data.",
        "No fitted drug parameters, expression profiles, PK-Sim individual overrides or simulation results copied.",
        "Repository nominal protocol and reference demographics require source-level caveats; they are not asserted to be original participant means.",
        "Age, sex, 70 kg or 175 cm defaults where unreported are explicit reference-adult assumptions."]
    if drug == "isoniazid":
        warnings += ["Both arms are labelled slow acetylators; no extrapolation to fast acetylators is claimed.",
                     "Development is an individual and target is a reported mean; aggregation/population sensitivity must be shown."]
    if drug == "sildenafil" and role != "calibration_data_reconstruction":
        warnings += ["Muirhead2002b explicit subject1 selected by lowest identifier, not curve outcome; individual-to-group differences are a sensitivity limitation.",
                     "FDA148-203 pooled individual observations not used as a mean trajectory or for NCA."]
    return {"compound_id": "pubchem:" + str(identity(drug)["pubchem_cid"]),
        "study_id": pick["study_id"], "role": role, "matrix": "plasma", "time_unit": "min", "unit": "umol/l",
        "molecular_weight_g_mol": mw, "human": human, "dosing": dosing, "observations": rows,
        "source": ref(record["path"], "ObservedData[" + observed["Name"] + "]"),
        "metadata_source": ref(record["path"], "Protocols[" + pick["protocol"] + "]; Individuals.OriginData only"),
        "clinical_reference_url": meta.get("Reference"), "original_observation_metadata": meta,
        "conversion": {"source_time_unit": grid["Unit"], "source_concentration_unit": column["Unit"],
            "time_multiplier": tf, "concentration_multiplier": factor,
            "parent_molecular_weight_g_mol": mw, "source_column_mw": column.get("DataInfo", {}).get("MolWeight")},
        "population": "slow_acetylator_adult" if drug == "isoniazid" else "reference_adult_with_source_limitations",
        "limitations": warnings}


def prepare():
    original, sources = read(CURRENT), read(SOURCE_INVENTORY)
    prior_inventory = read(BASE / "cohort_inventory.json") if (BASE / "cohort_inventory.json").exists() else None
    records = []
    for c in original["cases"]:
        records.append({"drug": c["drug"], "compound_id": c["compound_id"],
            "cohort_role": "existing_ten_development", "status": "existing_executable",
            "clinical_case": c, "physchem": ref(c["physchem"]),
            "independent_of_prior_algorithm_development": False})
    for drug in NAMES:
        old = LEGACY.get(drug)
        if drug == "furosemide":
            # Haegeli2007's inherited curve was already rejected by the project
            # source audit. Use another clinical study, never the rejected curve.
            dev = raw_curve(drug, RAW[drug][0], "calibration_data_reconstruction", sources)
            dev["clinical_reference_url"] = "https://pubmed.ncbi.nlm.nih.gov/6723758/"
            dev["limitations"].append("Haegeli2007 inherited digitization excluded because it contradicts the primary table; Hammarlund1984 healthy40mgIV substituted before any fitting.")
            req = read(OLD / old[0])  # only independent physical/1980 evidence
        elif old:
            dev, req = legacy_curve(drug, old[0], old[1], "calibration_data_reconstruction")
        else:
            dev, req = raw_curve(drug, RAW[drug][0], "calibration_data_reconstruction", sources), None
        if old and old[2]:
            target, _ = legacy_curve(drug, old[2], old[3], "historical_cross_study_reference")
        else:
            target = raw_curve(drug, RAW[drug][1], "historical_cross_study_reference", sources)
        if dev["study_id"] == target["study_id"] or dev["compound_id"] != target["compound_id"]:
            raise ValueError("Study or identity isolation failed")
        folder = BASE / "cohort" / drug
        dref, tref = save(folder / "development.json", dev), save(folder / "reference.json", target)
        if req:
            save(folder / "legacy_evidence_audit.json", {"request_source": ref(OLD / old[0]),
                "evidence": req["evidence"], "not_passed_as_unfiltered_designer_input": True,
                "clearance_policy": "Caffeine/alfentanil/sufentanil same-development CL replaced by new structure-only RF; independent CL retained for alprazolam/fluconazole/furosemide."})
        records.append({"drug": drug.capitalize(), "compound_id": dev["compound_id"],
            "cohort_role": "historical_extension", "status": "clinical_pair_prepared_requires_new_priors",
            "development": dref, "reference": tref, "identity": identity(drug),
            "independent_study_pair": True, "independent_of_prior_algorithm_development": False,
            "development_study_id": dev["study_id"], "target_study_id": target["study_id"],
            "development_point_count": len(dev["observations"]), "reference_point_count": len(target["observations"]),
            "development_dosing": dev["dosing"], "target_dosing": target["dosing"],
            "development_human": dev["human"], "target_human": target["human"],
            "reported_scalar_endpoints": "Not asserted. Original reported clinical scalars require separate curation; curve-based endpoints must be labelled observation-derived.",
            "sensitivity_flags": (["surgical_population"] if drug == "sufentanil" else
                ["slow_acetylator", "individual_to_group_mean"] if drug == "isoniazid" else
                ["group_mean_to_single_subject", "lowest_subject_id_prespecified"] if drug == "sildenafil" else
                ["pka_source_conflict_requires_explicit_choice"] if drug == "digoxin" else []),
            "newly_downloaded_clinical_observations": False,
            "raw_model_parameters_copied": False})
    result = {"version": "expanded-cohort-curation-1", "created_date": "2026-09-29",
        "target_unique_drugs": 20, "unique_drugs": len({r["compound_id"] for r in records}),
        "existing_drugs": 10, "additional_clinical_pairs": len(NAMES), "records": records,
        "source_pins": {str(CURRENT.relative_to(ROOT)): sha(CURRENT), str(SOURCE_INVENTORY.relative_to(ROOT)): sha(SOURCE_INVENTORY)},
        "excluded_candidates": [{"drug": "Propofol", "reason": "Whole-blood observations; plasma/blood equivalence not established"},
            {"drug": "Mexiletine", "reason": "AMI target and salt-to-base/protocol discrepancy; defer rather than pool with reference adults"},
            {"drug": "Amikacin", "reason": "Current source has one independent IV study; two subjects are not two studies"},
            {"drug": "Vancomycin", "reason": "Current chosen doses are from the same independent study"}],
        "policy": {"all_20_structure_and_scaffolds_excluded_from_RF": "pending_new_training",
            "five_or_three_point_fits_must_not_use_full_curve_NCA_CL": True,
            "do_not_require_all_eleven_references": True,
            "development_and_reference_documents_separate": True,
            "raw_model_sources_available_to_curator_only": True,
            "no_native_simulations_executed": True}}
    if (BASE / "priors/parameter_split.json").exists():
        result["priors"] = {"split": ref(BASE / "priors/parameter_split.json"),
            "predictions": ref(BASE / "priors/predictions.json"),
            "model": ref(BASE / "priors/parameter_models.joblib")}
        result["policy"]["all_20_structure_and_scaffolds_excluded_from_RF"] = True
    if prior_inventory and "additional_construction_ready" in prior_inventory:
        previous = {r["compound_id"]: r for r in prior_inventory["records"]}
        preserved = []
        for row in records:
            old = previous.get(row["compound_id"], {})
            if row.get("development") == old.get("development") and row.get("reference") == old.get("reference"):
                for key in ("status", "execution_ready", "native_simulation_validated", "physchem", "requests",
                            "independent_clearance_prior", "same_development_curve_CL_excluded", "new_rf_prior",
                            "new_rf_prior_available", "preflight", "blocked_reason", "conflict_source", "pka_curation_note"):
                    if key in old:
                        row[key] = old[key]
            preserved.append(row)
        result["records"] = preserved
        result["additional_ready_drugs"] = [r["drug"] for r in preserved if r.get("execution_ready")]
        result["additional_blocked_drugs"] = [r["drug"] for r in preserved if r.get("status", "").startswith("blocked_")]
        result["additional_construction_ready"] = len(result["additional_ready_drugs"])
        result["additional_blocked"] = len(result["additional_blocked_drugs"])
        result["interpretation"] = prior_inventory["interpretation"]
    save(BASE / "cohort_inventory.json", result)
    return result


def train_priors():
    import joblib
    from pbpk_kb.blueprints.pk_parameter_learning import run_parameter_pilot
    inventory = prepare()
    structures = {}
    for row in inventory["records"]:
        if row["cohort_role"] == "existing_ten_development":
            structures[row["drug"]] = read(ROOT / row["physchem"]["path"])["identity"]["isomeric_smiles"]
        else:
            structures[row["drug"]] = row["identity"]["isomeric_smiles"]
    folder = BASE / "priors"
    if (folder / "parameter_models.joblib").exists():
        split = read(folder / "parameter_split.json")
        if {n: r["input_smiles"] for n, r in split["external_exclusions"].items()} != structures:
            raise ValueError("Existing RF was trained with a different exclusion set")
        return prepare()
    folder.mkdir(exist_ok=True)
    learned = run_parameter_pilot(WHEEL, exclude_smiles=structures, train_limit=1283,
                                  test_limit=100, seed=20260929, n_jobs=4)
    predictions = {n: learned.predict(s) for n, s in structures.items()}
    if any(p["training_structure_overlap"] or p["training_scaffold_overlap"] for p in predictions.values()):
        raise ValueError("External-drug leakage in fitted RF")
    save(folder / "parameter_split.json", learned.audit)
    save(folder / "predictions.json", {"external_predictions": predictions, "held_out_parameter_metrics": learned.metrics,
        "source": ref(WHEEL), "clinical_curves_used_for_RF_training": False,
        "all_twenty_drug_structures_and_scaffolds_excluded": True})
    joblib.dump(learned.models, folder / "parameter_models.joblib", compress=3)
    return prepare()


def create_requests():
    """Make curated requests with new RF estimates, never full-curve NCA priors.

    Digoxin remains an explicit curation block because independent pKa records
    conflict. Its curve pair and RF prediction remain available for resolution.
    """
    from pbpk_kb.blueprints.evidence_blueprint import assemble_evidence_blueprint
    inventory = train_priors()
    estimates = read(BASE / "priors/predictions.json")["external_predictions"]
    prior_ref = ref(BASE / "priors/predictions.json", "external_predictions")
    chemical_method = {"name": "PubChem XLogP", "version": "2026-09-29 frozen response",
        "training_overlap": "unknown", "applicability": "Chemical descriptor, not a clinical endpoint"}
    rf_method = {"name": "random_forest", "version": "expanded20-20260929",
        "training_overlap": False, "applicability": {
            "target_structure_and_scaffold_excluded": True, "human_IV_parameter_model": True}}
    assumptions_path = BASE / "cohort" / "construction_assumptions.json"
    save(assumptions_path, {"solubility_mg_ml": 1., "solubility_ph": 7.,
        "binding_partner": "Unknown", "elimination_scope": "hepatic_only",
        "role": "Engineering IV-only placeholders and lumped clearance allocation; not measured evidence"})
    assumption_ref = ref(assumptions_path)
    ready, blocked = [], []
    for row in inventory["records"]:
        drug = row["drug"].lower()
        if row["cohort_role"] == "existing_ten_development":
            # Old ten remain attached to their existing protocol. Their RF
            # estimates must be replaced by the newly fitted exclusions before
            # combining cohorts in one new comparison.
            row["new_rf_prior_available"] = True
            row["new_rf_prior"] = prior_ref
            continue
        if drug == "digoxin":
            row.update(status="blocked_pka_source_conflict", execution_ready=False,
                blocked_reason="PubChem cites pKa 13.5 (Moj et al.) and acidic 7.15 (IARC/DrugBank). Do not choose by prediction error.",
                conflict_source=ref(BASE / "sources/digoxin_pka.json"))
            blocked.append(row["drug"])
            continue
        folder = BASE / "cohort" / drug
        dev, target = read(ROOT / row["development"]["path"]), read(ROOT / row["reference"]["path"])
        physical = row["identity"]
        original = read(OLD / LEGACY[drug][0]) if drug in LEGACY else None
        if original:
            evidence = deepcopy(original["evidence"])
            # These CL sources are from the same full development study and
            # would expose information unavailable to three/five-point fits.
            replace_cl = drug in {"caffeine", "alfentanil", "sufentanil"}
            evidence = [e for e in evidence if not (replace_cl and e["parameter"] == "total_clearance")]
        else:
            evidence = []

        def add(name, value, unit, basis, source, *, notes="", method=None, context=None):
            item = {"parameter": name, "compound_id": row["compound_id"], "value": value,
                "unit": unit, "basis": basis, "source": source, "context": context or {}, "notes": notes}
            if method is not None:
                item["prediction_method"] = method
            evidence.append(item)

        if original is None:
            add("molecular_weight", physical["molecular_weight_g_mol"], "g/mol", "derived", physical["source"])
            add("molecular_formula", physical["molecular_formula"], "text", "derived", physical["source"])
            add("logp", physical["xlogp"], "Log Units", "predicted", physical["source"], method=chemical_method)
            if drug == "triazolam":
                source = BASE / "sources/triazolam_pka_primary.json"
                if "1.52" not in source.read_text() or "6130140" not in source.read_text():
                    raise ValueError("Triazolam source does not support pKa")
                pka = [{"Type": "Base", "Pka": 1.52}]
                note = "Konishi 1982, PMID6130140: closed parent triazolam, not opened-form amine pKa6.50."
            elif drug == "sildenafil":
                source = BASE / "sources/sildenafil_pka_primary.xml"
                if "5.97" not in source.read_text() or "PMC10746450" not in source.read_text():
                    raise ValueError("Sildenafil source does not support pKa")
                pka = [{"Type": "Base", "Pka": 5.97}]
                note = "PMC10746450 Table1 compiles sildenafil pKa5.97; citrate salt MW is not used for parent-dose conversion."
            elif drug == "isoniazid":
                source = BASE / "sources/isoniazid_pka.json"
                if "pK1= 1.75; pK2= 3.57; pK3= 10.75" not in source.read_text():
                    raise ValueError("Isoniazid source does not support selected pKa sequence")
                pka = [{"Type": "Base", "Pka": 1.75}, {"Type": "Base", "Pka": 3.57},
                       {"Type": "Acid", "Pka": 10.75}]
                note = "IUPAC/Serjeant and Dempsey1979 sequence, recorded by PubChem HSDB; source-model fitted values not used."
            else:
                raise ValueError("Unreviewed new physicochemical source")
            add("pka_types", pka, "1", "derived", ref(source), notes=note)
            add("solubility", 1., "mg/ml", "assumption", assumption_ref,
                notes="Placeholder for already dissolved IV solution; not an observed solubility.")
            add("solubility_ph", 7., "1", "assumption", assumption_ref)
            add("binding_partner", "Unknown", "text", "assumption", assumption_ref)
            add("elimination_scope", "hepatic_only", "text", "assumption", assumption_ref,
                notes="Lumped total clearance; organ-specific elimination and DDI not validated.")
        vals = estimates[row["drug"]]["predictions"]
        present = {e["parameter"] for e in evidence}
        for parameter, endpoint, unit in (("total_clearance", "human_CL_mL_min_kg", "ml/min/kg"),
                                         ("fraction_unbound", "human_fup", "fraction")):
            if parameter not in present:
                add(parameter, vals[endpoint]["random_forest"], unit, "predicted", prior_ref, method=rf_method,
                    context={"species": "human", "matrix": "plasma", "route": "intravenous"},
                    notes="All20 compound structures and scaffold groups excluded; no clinical development/target curve used.")
        ledger = save(folder / "evidence.json", {"evidence": evidence})
        mw = next(e["value"] for e in evidence if e["parameter"] == "molecular_weight")
        physical_doc = {"identity": {k: v for k, v in physical.items() if k not in {"xlogp", "source"}},
            "source_files": [physical["source"]],
            "native_molecular_weight_g_mol": mw,
            "note": "Historical converted curves retain their previously verified parent-MW conventions; normalized native input MW listed separately."}
        # Scorers convert the native umol/l prediction with the same MW used in
        # construction; sub-0.1% source rounding differences are explicit.
        physical_doc["identity"]["molecular_weight_g_mol"] = mw
        physical_ref = save(folder / "physchem.json", physical_doc)
        request = {"entry_reason": "model_missing", "compound_id": row["compound_id"],
            "compound_name": row["drug"], "human": target["human"], "dosing": target["dosing"],
            "target": {"endpoints": ENDPOINTS, "primary_endpoint": "plasma_concentration_time",
                "window_start_h": 0., "window_end_h": min(168., max(24., math.ceil(target["observations"][-1]["time_min"] / 60)))},
            "partition_method": "PKSimStandard", "evidence_ledger": ledger, "evidence": evidence,
            "forbidden_sources": [row["reference"]],
            "assumptions": ["Historical cross-study development comparison; not a new blind test.",
                "Clinical reference outcomes cannot be used to choose parameters or formulas.",
                "Same-development CL summaries replaced by RF; independent third-study CL retained only where documented.",
                "RF excludes all20 target structures/scaffolds; measured inputs are not RF outputs.",
                "Representative-human assumptions retained from curated study metadata.",
                "No organ-concentration, disease, genotype-mechanism or DDI validation is claimed."]}
        development_request = deepcopy(request)
        development_request["human"], development_request["dosing"] = dev["human"], dev["dosing"]
        development_request["target"]["window_end_h"] = max(request["target"]["window_end_h"],
            math.ceil(dev["observations"][-1]["time_min"] / 60))
        status = []
        for req in (development_request, request):
            assembled = assemble_evidence_blueprint(req, ROOT)
            status.append(assembled["status"])
            if assembled["status"] != "assembled":
                raise ValueError("Construction preflight rejected " + drug + ": " + str(assembled))
        drequest = save(folder / "development_request.json", development_request)
        trequest = save(folder / "target_request.json", request)
        row.update(status="construction_preflight_passed", execution_ready=True,
            native_simulation_validated=False, physchem=physical_ref,
            requests={"development": drequest, "target": trequest},
            independent_clearance_prior=(drug in {"alprazolam", "fluconazole", "furosemide"}),
            same_development_curve_CL_excluded=True,
            new_rf_prior=prior_ref,
            preflight={"development": status[0], "target": status[1]})
        ready.append(row["drug"])
    inventory.update(additional_construction_ready=len(ready), additional_blocked=len(blocked),
        additional_ready_drugs=ready, additional_blocked_drugs=blocked,
        interpretation="Prepared clinical pairs are not native-execution successes; requests passed structural assembly only. Ten original cases must refresh RF inputs before a unified20-drug comparison.")
    save(BASE / "cohort_inventory.json", inventory)
    return inventory


def verify_prepared():
    from pbpk_kb.blueprints.development_curve_blueprint import _training_document
    from pbpk_kb.blueprints.evidence_blueprint import assemble_evidence_blueprint
    inventory = read(BASE / "cohort_inventory.json")
    split = read(BASE / "priors/parameter_split.json")
    external = list(split["external_exclusions"].values())
    identities = {r["identity_smiles"] for r in external}
    scaffolds = {r["scaffold_group"] for r in external}
    assert len(external) == inventory["unique_drugs"] == 20
    assert not set(split["train_indices"]) & set(split["test_indices"])
    for r in split["rows"]:
        if r["status"] in {"train", "test"}:
            assert r["identity_smiles"] not in identities
            assert r["scaffold_group"] not in scaffolds
    verified = 0
    for row in inventory["records"]:
        if not row.get("execution_ready"):
            continue
        for source in [row["development"], row["reference"], *row["requests"].values()]:
            assert sha(source["path"]) == source["sha256"], source["path"]
        dev, target = read(ROOT / row["development"]["path"]), read(ROOT / row["reference"]["path"])
        assert dev["study_id"] != target["study_id"]
        assert dev["compound_id"] == target["compound_id"] == row["compound_id"]
        for request_ref in row["requests"].values():
            req = read(ROOT / request_ref["path"])
            evidence_sources(req)
            assert assemble_evidence_blueprint(req, ROOT)["status"] == "assembled"
        req = read(ROOT / row["requests"]["development"]["path"])
        _training_document({**row["development"], "role": "calibration_data_reconstruction"}, req, ROOT)
        if row["drug"] in {"Caffeine", "Alfentanil", "Sufentanil"}:
            cl = next(e for e in req["evidence"] if e["parameter"] == "total_clearance")
            assert cl["basis"] == "predicted" and cl["prediction_method"]["version"] == "expanded20-20260929"
        verified += 1
    print(json.dumps({"verified_additional_request_pairs": verified, "independent_study_pairs": verified,
        "excluded_RF_compounds": len(external), "RF_train": len(split["train_indices"]),
        "RF_test": len(split["test_indices"]), "clinical_curve_labels_used_for_RF": False,
        "original_reported_scalar_endpoints_not_asserted": True,
        "native_simulations_executed": False, "blocked": inventory["additional_blocked_drugs"]}))
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train-priors", "requests", "verify"])
    args = parser.parse_args()
    result = verify_prepared() if args.command == "verify" else create_requests() if args.command == "requests" else train_priors() if args.command == "train-priors" else prepare()
    print(json.dumps({"unique_drugs": result["unique_drugs"],
        "additional_clinical_pairs": result["additional_clinical_pairs"],
        "rf_priors_ready": bool(result.get("priors")), "inventory": str((BASE / "cohort_inventory.json").relative_to(ROOT))}))


if __name__ == "__main__":
    main()
