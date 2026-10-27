#!/usr/bin/env python3
"""Reproduce independent chemistry from three frozen, bounded source files.

No network, archived Compound blocks, clinical curves or outcome labels are
opened. Choices below are source-based and fixed before the new predictions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BATCH = "experiments/blueprint_design/expanded_validation_20261007"
VERSION = "expanded-independent-chemical-inputs-1"
DRUGS = ("Amikacin", "Clarithromycin", "Ethinylestradiol", "Mexiletine",
         "Montelukast", "Oxycodone", "Paroxetine", "Pitavastatin", "Probenecid",
         "Rifampicin", "Rosuvastatin", "Vancomycin")
PKA_PATH = "/Record/Section/0/Section/0/Section/0/Information/"
BINDING_PATH = "/Record/Section/0/Section/0/Information/0"
PKA = {
    "Clarithromycin": (0, "8.99", [{"Type": "Base", "Pka": 8.99}]),
    "Mexiletine": (0, "9.14", [{"Type": "Base", "Pka": 9.14}]),
    "Oxycodone": (0, "8.53", [{"Type": "Base", "Pka": 8.53}]),
    "Paroxetine": (0, "9.51", [{"Type": "Base", "Pka": 9.51}]),
    "Probenecid": (0, "3.01", [{"Type": "Acid", "Pka": 3.01}]),
    "Rifampicin": (1, "Zwitterion with pKa 1.7 related to the 4-hydroxy and pKa 7.9 related to the 3-piperazine nitrogen",
                   [{"Type": "Acid", "Pka": 1.7}, {"Type": "Base", "Pka": 7.9}]),
}
# Precise source assertions prevent a generic numeric parser from converting
# ranges, inequalities or a different annotation into a false point value.
BINDING = {
    "Clarithromycin": ("~ 70% protein bound", 70., True),
    "Oxycodone": ("45%. Oxycodone is primarily bound to serum albumin and to a lesser degree alpha1-acid glycoprotein.", 45., False),
    "Paroxetine": ("Paroxetine is 95% bound to plasma proteins.", 95., False),
    "Rifampicin": ("Rifampin is about 80% protein bound. It may bind to serum albumin. Most of the unbound fraction is not ionized and, therefore, diffuses freely into tissues.", 80., True),
    "Rosuvastatin": ("Rosuvastatin is 88% bound to plasma proteins, mostly albumin. This binding is reversible and independent of plasma concentrations.", 88., False),
}
MULTISITE = {
    "Amikacin": (1, "pKa1 = 6.7 (primary amine); pKa2 = 8.4 (primary amine); pKa3 = 8.4 (primary amine); pKa4 = 9.7 (primary amine)", [6.7, 8.4, 8.4, 9.7]),
    "Vancomycin": (1, "pka1 = 2.6; pKa2 = 7.2; pKa3 = 8.6; pKa4 = 9.6; pKa5 = 10.5; pKa6 = 11.7", [2.6, 7.2, 8.6, 9.6, 10.5, 11.7]),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, path):
    root = Path(root).resolve(strict=True)
    path = (root / path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Source/output must stay in the project")
    return path


def pointer(value, path):
    for token in path.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def words(info):
    return " ".join(x["String"] for x in info["Value"]["StringWithMarkup"])


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ref(root, path, location=None):
    result = {"path": str(path.relative_to(root)), "sha256": sha(path)}
    if location is not None:
        result["location"] = location
    return result


def evidence(cid, parameter, value, unit, basis, source, notes, context=None):
    result = {"parameter": parameter, "compound_id": cid, "value": value,
              "unit": unit, "basis": basis, "source": source, "notes": notes,
              "context": {"target_observations_used": False, "fitted_compound_parameters_used": False,
                          **(context or {})}}
    if basis == "predicted":
        result["prediction_method"] = {"name": "PubChem XLogP" if parameter == "logp" else "ChemAxon pKa",
                                       "version": "unreported_by_frozen_source",
                                       "training_overlap": "unknown",
                                       "applicability": {"scope": "structure_based_physicochemical_estimation",
                                                         "target_clinical_curve_used_for_calibration": False,
                                                         "clinical_predictive_validity_established": False},
                                       "parameter_prediction_only": True}
    return result


def annotation(data, drug, heading, location):
    record = data["records"][drug + "|" + heading]
    if record["status"] != "downloaded":
        raise ValueError("Required frozen annotation unavailable: " + drug + "/" + heading)
    return record, pointer(record["response"], location)


def make_artifact(project_root=ROOT):
    root = Path(project_root).resolve(strict=True)
    paths = {key: inside(root, BATCH + "/" + file) for key, file in (
        ("properties", "pubchem_properties.json"), ("annotations", "pubchem_annotations.json"),
        ("literature", "chemical_literature_sources.json"))}
    sources = {key: json.loads(path.read_text()) for key, path in paths.items()}
    initial_hashes = {key: sha(path) for key, path in paths.items()}
    compounds = {}
    for drug in DRUGS:
        record = sources["properties"]["records"][drug]
        if record["status"] != "downloaded":
            raise ValueError("Properties unavailable for " + drug)
        entries = record["response"]["PropertyTable"]["Properties"]
        if len(entries) != 1:
            raise ValueError("Ambiguous molecular identity: " + drug)
        props = entries[0]
        cid = "pubchem:" + str(int(props["CID"]))
        mw, smiles = float(props["MolecularWeight"]), props["SMILES"]
        if not finite(mw) or mw <= 0 or not isinstance(smiles, str) or not smiles or "." in smiles:
            raise ValueError("A single parent structure and positive molecular weight are required")
        formula = props["MolecularFormula"]
        if not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", formula) or not finite(props["XLogP"]):
            raise ValueError("Malformed formula or XLogP: " + drug)
        base = "/records/" + drug + "/response/PropertyTable/Properties/0/"
        rows = []
        for parameter, value, unit, field, basis in (
            ("molecular_weight", mw, "g/mol", "MolecularWeight", "derived"),
            ("molecular_formula", formula, "text", "MolecularFormula", "derived"),
            ("logp", float(props["XLogP"]), "Log Units", "XLogP", "predicted")):
            rows.append(evidence(cid, parameter, value, unit, basis, ref(root, paths["properties"], base + field),
                "PubChem parent-structure property. XLogP is computed, not a clinical measurement or a PBPK-fitted logP.",
                {"url": record["url"], "pubchem_cid": props["CID"], "parent_molecule": True}))
        missing, exclusions, independent = [], [], []
        if drug in PKA:
            index, expected, pka = PKA[drug]
            loc = PKA_PATH + str(index)
            ann, info = annotation(sources["annotations"], drug, "Dissociation Constants", loc)
            if words(info) != expected:
                raise ValueError("Reviewed pKa annotation changed: " + drug)
            rows.append(evidence(cid, "pka_types", deepcopy(pka), "1", "measured",
                ref(root, paths["annotations"], "/records/" + drug + "|Dissociation Constants/response" + loc),
                "Literature-reported pKa from the explicitly selected PubChem annotation. Other sources may differ; no value was selected using prediction outcomes.",
                {"url": ann["url"], "annotation_reference_number": info["ReferenceNumber"],
                 "original_references": info.get("ExtendedReference", []), "experimental_measurement_independently_repeated": False}))
        elif drug in MULTISITE:
            index, expected, values = MULTISITE[drug]
            ann, info = annotation(sources["annotations"], drug, "Dissociation Constants", PKA_PATH + str(index))
            if words(info) != expected:
                raise ValueError("Reviewed multisite pKa annotation changed: " + drug)
            missing.append(f"{len(values)} reported ionization constants exceed the current maximum of 3; no independently validated reduced representation is established")
            exclusions.append({"parameter": "pka_types", "reported_pka_values": values,
                "source": ref(root, paths["annotations"], "/records/" + drug + "|Dissociation Constants/response" + PKA_PATH + str(index)),
                "reason": "Do not truncate sites or merge equal pKa values to force a supported model"})
        else:
            fact = sources["literature"]["records"][drug]
            pka = fact["pka_types"]
            if not 0 < len(pka) <= 3 or any(p["Type"] not in {"Acid", "Base"} or not finite(p["Pka"]) for p in pka):
                raise ValueError("Unsupported external pKa representation: " + drug)
            rows.append(evidence(cid, "pka_types", deepcopy(pka), "1", fact["pka_basis"],
                ref(root, paths["literature"], "/records/" + drug + "/pka_types"), fact["pka_notes"],
                {"url": fact["url"], "citation": fact["citation"], "source_location": fact["source_location"],
                 "representation_scope": "Published dominant/macro pKa values; not a complete microscopic ionization model"}))
            exclusions.extend(deepcopy(fact.get("excluded_facts", [])))
        ann, info = annotation(sources["annotations"], drug, "Protein Binding", BINDING_PATH)
        annref = ref(root, paths["annotations"], "/records/" + drug + "|Protein Binding/response" + BINDING_PATH)
        if drug in BINDING:
            expected, bound, approximate = BINDING[drug]
            if words(info) != expected:
                raise ValueError("Reviewed protein binding statement changed: " + drug)
            independent.append(evidence(cid, "fraction_unbound", round(1 - bound / 100, 8), "fraction", "derived", annref,
                "fu = 1 - reported bound percentage / 100. General human drug reference, not a patient-specific measurement. Source is separate from the target curve file; underlying review-study overlap is unknown.",
                {"species": "human", "matrix": "plasma", "url": ann["url"], "bound_percent": bound,
                 "approximate_reported_value": approximate, "general_protein_binding_adopted_for_plasma": drug in {"Clarithromycin", "Oxycodone", "Rifampicin"},
                 "underlying_study_overlap": "unknown", "evidence_role": "independent_general_drug_binding", "target_curve_source_used": False}))
        else:
            exclusions.append({"parameter": "fraction_unbound", "source": annref,
                "reason": "No selected unambiguous human-plasma point value: interval/inequality, matrix ambiguity, or chemically blocked request. Do not turn bounds into a point; permit declared ML fallback."})
        compounds[drug] = {"compound_id": cid, "smiles": smiles, "molecular_weight_g_mol": mw,
            "evidence": rows, "independent_evidence": independent, "construction_status": "blocked" if missing else "ready",
            "missing_reasons": missing, "excluded_evidence": exclusions,
            "interpretation": "Ready means independent chemical inputs are present; it is not a prediction or validation success."}
    if any(sha(path) != initial_hashes[key] for key, path in paths.items()):
        raise ValueError("Frozen source changed during chemical curation")
    return {"version": VERSION,
        "source_refs": {str(path.relative_to(root)): initial_hashes[key] for key, path in paths.items()},
        "code_pins": {str(Path(__file__).resolve().relative_to(root)): sha(__file__)}, "compounds": compounds,
        "counts": {"chemical_reviewed_compounds": len(compounds), "chemical_ready": sum(c["construction_status"] == "ready" for c in compounds.values()),
                   "chemical_blocked": sum(c["construction_status"] == "blocked" for c in compounds.values()),
                   "with_independent_fu": sum(bool(c["independent_evidence"]) for c in compounds.values())},
        "construction_rule": "Independent structural facts and explicit predicted XLogP; selected pKa with provenance; no OSP fitted compound parameters or target clinical endpoints.",
        "limitations": ["No clinical CL, Vss, half-life or MRT is imported from the target studies.",
            "Binding inputs from general references are approximate; complete underlying-study independence is unverified.",
            "Predicted or interpreted pKa values are not presented as new experimental measurements.",
            "Chemical availability does not grant clinical cohort eligibility. The cohort artifact controls membership; Oxycodone chemistry is retained although its clinical study arm and parent-dose interpretation remain pending.",
            "Unsupported multisite drugs remain in the clinical-attempt denominator; clinically pending drugs are not promoted by chemical availability."],
        "target_observations_read": False, "osp_compound_parameters_used": False, "network_requests_during_regeneration": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing artifact byte-for-byte without writing")
    args = parser.parse_args()
    result = make_artifact()
    path = inside(ROOT, BATCH + "/chemical_sources.json")
    text = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError("Existing chemical artifact differs; refusing to overwrite a possible frozen input")
    elif args.check:
        raise FileNotFoundError(path)
    else:
        path.write_text(text)
    print(json.dumps({"path": str(path), "sha256": sha(path), **result["counts"]}))


if __name__ == "__main__":
    main()
