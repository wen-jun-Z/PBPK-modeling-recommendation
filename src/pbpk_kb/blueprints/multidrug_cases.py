"""Locked adult GFR cases with external observations separated from design input.

This data adapter prepares evaluation records but never fits or selects a model.
The returned requests contain drug/person/dose inputs, no PK observations, old
simulation parameters or fitted structures. Pregabalin retains a fitted oral
permeability solely as an explicitly labelled reconstruction control. Other
cases use literature inputs; this does not make their source studies newly blind.
An independent evaluator must check prediction locking before opening holdouts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

from .development_case import DevelopmentCaseError, _one, _plain, _reference, _verify
from .human_model import load_standard_human_model

_COLLECTION = "artifacts/pbpk-model-assets/public_collection_20260909/"
_IDENTITY = ("experiments/blueprint_design/development_inputs/multidrug_sources/pubchem_identity_facts_20260918.json",
             "93ece0d027ff170eae374d2e820c38016c456cfb08bc42f4d5bd266a553ea835")
_VANCO_LITERATURE = (
    _COLLECTION + "Vancomycin-Model__155fdca3eb25/source/Evaluation/Input/Content/Section2.2_Data_used.md",
    "4b3f4d66016d4871883cf35303da9c8b527c0d8438d8ef5dcb320e28d297ef37")
_REGISTRY = {
    "amikacin": {
        "source": (_COLLECTION + "Amikacin-Model__c2e9c7a4e0ca/source/Amikacin-Model.json",
                   "5ee48d408a5587feed2f0c6181c8f2b1e9a77a6ad7a4ba70f261e84f526eecd9"),
        "name": "Amikacin", "study": "Walker 1979", "tier": "bottom_up_external",
        "simulations": ["Walker 1979 Volunteer 6", "Walker 1979 Volunteer 9"],
        "limitations": ["Only one independent study; individual volunteers are not independent studies.",
                        "Snapshot MW 588.6 conflicts with PubChem parent MW 585.6; verified parent MW is used in both input and concentration conversion.",
                        "Published fu=1 and logP=-0.48 are inherited literature assumptions, not new measurements."],
    },
    "vancomycin": {
        "source": (_COLLECTION + "Vancomycin-Model__155fdca3eb25/source/Vancomycin.json",
                   "ad435c0ea810f66c7c075b749b4e8b92b5c3dd5c9a899e3cb7cc1522ec3aef5a"),
        "name": "Vancomycin", "study": "Boekh 1988", "tier": "bottom_up_external",
        "simulations": ["Boeckh 1988 Figure 1 0.5g", "Boeckh1988_1g"],
        "limitations": ["Boekh/Boeckh 1988 spelling variants denote one study; two doses are not independent studies.",
                        "Fitted snapshot logP/fu are replaced with cited literature-table DrugBank logP=1.11 and Zhou2016 fu=0.48 before any scoring.",
                        "GFR-only elimination is an explicit hypothesis; published alternatives include additional hepatic elimination.",
                        "Vancomycin is a glycopeptide represented by PK-Sim's small-molecule model, not a conventional low-molecular-weight drug.",
                        "Observed source incorrectly names amikacin in formulation metadata; molecule/dose/protocol all identify vancomycin. This discrepancy remains disclosed."],
    },
    "pregabalin": {
        "source": (_COLLECTION + "Pregabalin-Model__47b22b0be8cd/source/Pregabalin.json",
                   "0a4fbffe1e6adae54c58344ced780be8eff0a2547a099d80cf3e3ebd2e23588f"),
        "name": "Pregabalin", "study": "Bockbrader 2010", "tier": "reconstruction",
        "simulations": ["Bockbrader2010_1mg", "Bockbrader2010_2mg"],
        "limitations": ["Oral permeability remains an openly fitted historical value; this group is reconstruction only and cannot establish independent design quality.",
                        "Snapshot fitted logP is replaced with computed PubChem XLogP=-1.6; this is computed, not experimentally measured.",
                        "Only the adult solution arms are used; capsule/controlled-release and pediatric studies are excluded.",
                        "Solubility/reference pH and oral water volume include source assumptions."],
    },
}


def _write_once(path: Path, document: dict) -> str:
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode()
    checksum = hashlib.sha256(encoded).hexdigest()
    if path.is_symlink():
        raise DevelopmentCaseError("Multidrug output must not be a symlink.")
    if path.exists():
        if path.read_bytes() != encoded:
            raise DevelopmentCaseError("Refusing to overwrite a changed multidrug input.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(encoded)
    return checksum


def _metadata(observed: dict) -> dict:
    return {item["Name"]: item.get("Value") for item in observed.get("ExtendedProperties", [])}


def _external_observations(observed: dict, molecular_weight: float) -> list[dict]:
    """Data-preparation-only conversion; caller writes a separate evaluator file."""
    grid = observed["BaseGrid"]
    columns = [column for column in observed["Columns"] if column.get("DataInfo", {}).get("Origin") == "Observation"]
    if len(columns) != 1:
        raise DevelopmentCaseError("Expected one externally observed concentration column.")
    column = columns[0]
    time_factors = {"h": 60.0, "min": 1.0, "s": 1 / 60}
    unit = column["Unit"].replace("µ", "u").replace("μ", "u").lower()
    factors = {"umol/l": 1.0, "mg/l": 1000 / molecular_weight,
               "ug/ml": 1000 / molecular_weight, "ug/l": 1 / molecular_weight,
               "ng/ml": 1 / molecular_weight}
    if grid["Unit"] not in time_factors or unit not in factors:
        raise DevelopmentCaseError("Unsupported external observation units.")
    if len(grid["Values"]) != len(column["Values"]) or not grid["Values"]:
        raise DevelopmentCaseError("External observation columns have incompatible lengths.")
    result = []
    for index, (time, value) in enumerate(zip(grid["Values"], column["Values"], strict=True)):
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (time, value)) or time < 0 or value <= 0:
            raise DevelopmentCaseError("External curve contains invalid or nonpositive concentration; do not silently drop points.")
        result.append({"observation_id": f"external_{index}", "time_min": time * time_factors[grid["Unit"]],
                       "concentration_umol_per_l": value * factors[unit], "unit": "umol/l"})
    if any(a["time_min"] >= b["time_min"] for a, b in zip(result, result[1:])):
        raise DevelopmentCaseError("External observation times must be strictly increasing.")
    return result


def _make_case(root: Path, output: Path, key: str, definition: dict, identities: dict) -> dict:
    registration = _REGISTRY[key]
    source = registration["source"]
    snapshot = _verify(root, *source, parse=True)
    identity = identities["compounds"][key]
    compound = _plain(_one(snapshot["Compounds"], "Name", registration["name"]))
    compound["CalculationMethods"] = ["Cellular partition coefficient method - PK-Sim Standard", "Cellular permeability - PK-Sim Standard"]
    compound["Description"] = "Source-locked adult development reference; no observed data or old simulation structure copied."
    formula = identity["MolecularFormula"]
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*)", formula)
    if "".join(atom + count for atom, count in tokens) != formula:
        raise DevelopmentCaseError("Unsupported molecular formula notation.")
    counts = {atom: int(count or 1) for atom, count in tokens}
    mw = _one(compound["Parameters"], "Name", "Molecular weight")
    if "MolecularWeight" in identity:
        mw["Value"] = identity["MolecularWeight"]
    compound["Parameters"] = [mw, *({"Name": atom, "Value": counts.get(atom, 0)} for atom in ("F", "Cl", "Br", "I"))]
    solubility = _one(compound["Solubility"][0]["Parameters"], "Name", "Solubility at reference pH")
    if solubility["Unit"] != "mg/l":
        raise DevelopmentCaseError("Unexpected source solubility unit.")
    solubility.update(Value=solubility["Value"] / 1000, Unit="mg/ml")
    overrides = []
    if key == "vancomycin":
        _verify(root, *_VANCO_LITERATURE)
        compound["Lipophilicity"][0]["Parameters"][0]["Value"] = 1.11
        compound["FractionUnbound"][0]["Parameters"][0]["Value"] = .48
        overrides = ["Lipophilicity", "FractionUnbound"]
    if key == "pregabalin":
        compound["Lipophilicity"][0]["Parameters"][0]["Value"] = identity["XLogP"]
        permeability = compound["IntestinalPermeability"][0]["Parameters"][0]
        if permeability["Unit"] != "cm/min":
            raise DevelopmentCaseError("Unexpected reconstruction permeability unit.")
        permeability.update(Value=permeability["Value"] / 60, Unit="cm/s")
        compound["IntestinalPermeability"][0]["IsDefault"] = True
    if len(compound["Processes"]) != 1 or compound["Processes"][0]["InternalName"] != "GlomerularFiltration":
        raise DevelopmentCaseError("Only the verified single GFR process is supported.")
    sources = {field: _reference(*source, f"Compounds[{registration['name']}].{field}; {registration['tier']}")
               for field in compound if field not in {"Name", "Description", "IsSmallMolecule"}}
    sources["Parameters"] = _reference(*_IDENTITY, f"compounds.{key}; formula-derived halogens; MW from identity facts if present, otherwise source snapshot")
    for field in overrides:
        sources[field] = _reference(*_VANCO_LITERATURE, "Section2.2.1 literature table; fixed before evaluation")
    if key == "pregabalin":
        sources["Lipophilicity"] = _reference(*_IDENTITY, "compounds.pregabalin.XLogP; computed, not fitted")
    scenarios = []
    for index, simulation_name in enumerate(registration["simulations"]):
        simulation = _one(snapshot["Simulations"], "Name", simulation_name)
        individual = _plain(_one(snapshot["Individuals"], "Name", simulation["Individual"]))
        individual["OriginData"].pop("CalculationMethods", None)
        if not individual.get("ExpressionProfiles"):
            individual.pop("ExpressionProfiles", None)
        if individual["OriginData"]["Species"] != "Human" or individual["OriginData"]["Age"]["Value"] < 18:
            raise DevelopmentCaseError("Only sourced adult human individuals are admitted.")
        protocol_binding = simulation["Compounds"][0]["Protocol"]
        protocol = _plain(_one(snapshot["Protocols"], "Name", protocol_binding["Name"]))
        if protocol["DosingInterval"] != "Single":
            raise DevelopmentCaseError("Only selected single-dose scenarios are supported.")
        oral = protocol["ApplicationType"] == "Oral"
        formulation = None
        if oral:
            formulation = _plain(_one(snapshot["Formulations"], "Name", protocol_binding["Formulations"][0]["Name"]))
            if formulation["FormulationType"] != "Formulation_Dissolved":
                raise DevelopmentCaseError("Only dissolved oral formulations are admitted.")
            formulation["Parameters"] = []
        dose = _one(protocol["Parameters"], "Name", "InputDose")
        target = f"Organism|PeripheralVenousBlood|{registration['name']}|Plasma (Peripheral Venous Blood)"
        instance = {
            "population_context": "healthy_adult", "individual": individual,
            "generated_demographics": [name for name in ("Weight", "Height") if name not in individual["OriginData"]],
            "compound": copy.deepcopy(compound), "protocol": protocol, "snapshot_version": 80,
            "simulation_name": f"{registration['name']}_External_{index + 1}",
            "human_source": _reference(*source, f"Individuals[{individual['Name']}]; unreported demographics generated by PK-Sim"),
            "compound_sources": copy.deepcopy(sources),
            "protocol_source": _reference(*source, f"Protocols[{protocol['Name']}]; original dose and infusion; oral water is an assumption"),
            "formulation": formulation,
            "formulation_source": _reference(*source, f"Formulations[{formulation['Name']}]") if oral else None,
            "expression_profiles": [], "expression_profile_sources": {},
            "output_schema": _plain(simulation["OutputSchema"]), "output_selections": [target],
        }
        requirement = {"compound_semantic_id": f"pubchem:{identity['CID']}", "name": registration["name"],
                       "population": "healthy_adult", "route": "oral" if oral else "intravenous",
                       "dose": dose["Value"], "dose_unit": dose["Unit"],
                       "formulation": formulation["Name"] if oral else None,
                       "targets": [{"target_id": key + "_parent_plasma", "analyte": "parent", "matrix": "plasma",
                                    "quantity": "concentration", "binding": "total", "required": True}]}
        request = {"requirement": requirement, "human_model": {"registry_id": definition["registry_id"], "definition_checksum": definition["checksum"]},
                   "instance": instance, "base": None, "modules": [], "evidence": [], "studies": []}
        observed = _one(snapshot["ObservedData"], "Name", simulation["ObservedData"][0])
        meta = _metadata(observed)
        if meta.get("Study Id") != registration["study"] or meta.get("Molecule") != registration["name"] or meta.get("Species") != "Human":
            raise DevelopmentCaseError("External observation study, species or compound identity mismatch.")
        scenario_id, study_id = f"{key}_{index + 1}", key + "_" + re.sub(r"[^a-z0-9]+", "_", registration["study"].lower()).strip("_")
        external = {"schema_version": "multidrug-external-observations-1.0", "scenario_id": scenario_id,
                    "compound_id": f"pubchem:{identity['CID']}",
                    "study_group_id": study_id, "source_study_id": registration["study"], "purpose": "external_reference_only",
                    "tier": registration["tier"], "source": _reference(*source, f"ObservedData[{observed['Name']}]"),
                    "metadata": meta, "unit": "umol/l", "time_unit": "min",
                    "molecular_weight_used_g_per_mol": mw["Value"],
                    "observations": _external_observations(observed, mw["Value"])}
        path = output / "external_observations" / (scenario_id + ".json")
        checksum = _write_once(path, external)
        scenarios.append({"scenario_id": scenario_id, "study_group_id": study_id, "role": "external",
                          "design_request": request, "target_output": target, "unit": "umol/l", "time_unit": "min",
                          "observation_count": len(external["observations"]),
                          "holdout_reference": {"path": path.relative_to(root).as_posix(), "sha256": checksum},
                          "source_study_id": registration["study"], "source_observation_name": observed["Name"]})
    return {
        "case_id": key + "_adult_fixed_gfr", "compound_id": f"pubchem:{identity['CID']}", "compound_name": registration["name"],
        "purpose": "historical_development_external_reference", "tier": registration["tier"],
        "strict_main_eligible": False, "design_request": copy.deepcopy(scenarios[0]["design_request"]),
        "requirement": copy.deepcopy(scenarios[0]["design_request"]["requirement"]), "scenarios": scenarios,
        "parameter_search": {"parameters": [], "fit_permitted": False, "selection_permitted": False},
        "source_audit": [{"path": source[0], "sha256": source[1]}, {"path": _IDENTITY[0], "sha256": _IDENTITY[1]}]
                        + ([{"path": _VANCO_LITERATURE[0], "sha256": _VANCO_LITERATURE[1]}] if overrides else []),
        "limitations": registration["limitations"] + [
            "All chosen scenarios belong to one study per drug; no Dcal/Dsel split is fabricated.",
            "This is a locked fixed-input comparison of native constitutive rules; external errors may not tune or select the model.",
            "Source reports informed historical model development; these are not newly blinded independent benchmark studies.",
            "The evaluator must verify the prediction lock before reading external observation files.",
            "Standard-model equations and human physiology are regenerated; source simulation Parameters, Kp values and results are not copied.",
            "Unreported height is generated from PK-Sim; source population/gender/age/weight are explicit.",
        ],
    }


def prepare_multidrug_inputs(project_root: Path, output_dir: Path, *, compounds: list[str] | None = None) -> dict:
    """Write evaluator-only observations and return observation-free requests.

    Output must be a new or byte-identical project-contained directory. Sources
    are hash pinned. This is data preparation, never external-score-driven model
    design. Unknown compounds fail rather than silently substituting a dataset.
    """
    root = Path(project_root).resolve()
    output = Path(output_dir)
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    if not output.is_relative_to(root) or output == root:
        raise DevelopmentCaseError("Multidrug output must remain inside the project.")
    keys = list(_REGISTRY) if compounds is None else list(compounds)
    if not keys or len(set(keys)) != len(keys) or any(key not in _REGISTRY for key in keys):
        raise DevelopmentCaseError("Unknown or duplicate multidrug case.")
    identities = _verify(root, *_IDENTITY, parse=True)
    definition = load_standard_human_model(root)
    output.mkdir(parents=True, exist_ok=True)
    cases = [_make_case(root, output, key, definition, identities) for key in keys]
    manifest = {"schema_version": "multidrug-fixed-inputs-1.0", "cases": cases,
                "exclusions": [
                    {"compound": "Inulin", "reason": "Downloaded source individuals are Rat, not Human."},
                    {"compound": "Pregabalin", "scope": "strict_main", "reason": "Original oral permeability was fitted; current case is reconstruction only."},
                    {"compound": "All included drugs", "scope": "M0-M6 strict_main", "reason": "No independent calibration/selection/test study split is established by this fixed-input package."},
                ], "external_observations_in_design_payload": False, "solver_calls": 0}
    _write_once(output / "input_manifest.json", manifest)
    return manifest


_ALFENTANIL_SOURCE = (
    _COLLECTION + "Alfentanil-Model__843af5794569/source/Alfentanil-Model.json",
    "783a1fcd05eb89614ec33cb580886383bb7e8cd8f532fea851a9ab41326d0ceb")
_ALFENTANIL_LITERATURE = (
    _COLLECTION + "Alfentanil-Model__843af5794569/source/Evaluation/Input/Content/Section2.2_Data.md",
    "17fd4a27671f13f5fbf62b780592a99a7a95ef23090e2bd6776a4c4eea35f3cd")
_ALFENTANIL_IDENTITY = (
    "experiments/blueprint_design/development_inputs/multidrug_sources/alfentanil_identity_facts_20260918.json",
    "3c1e3b62d6c082b012aae384fa4376ed039fd708197f17a3076845214f746d25")
_ALFENTANIL_STUDIES = (
    ("Kharasch 1997", "calibration", "Kharasch 1997, Alfentanil iv 0.02 mg/kg", 14),
    ("Kharasch 2004", "calibration", "Kharasch 2004, Alfentanil iv 0.015 mg/kg", 13),
    ("Phimmasone 2001", "selection", "Phimmasone 2001, Alfentanil iv 0.015 mg/kg", 16),
    ("Kharasch 2011", "external", "Kharasch 2011, Alfentanil iv 0.015 mg/kg", 15),
)


def prepare_alfentanil_development_inputs(project_root: Path, output_dir: Path) -> dict:
    """Prepare a four-study historical development split, separate final curves.

    Two IV studies calibrate, a third selects and a fourth is exported only for
    the independent evaluator. All four were used by the historical public
    model, so this cannot be called a new blind main benchmark. No old optimum,
    organ permeability, simulation overrides or CYP3A4 expression fit is copied.
    The two numerical search starts/bounds are the public pre-fit configuration.
    """
    root = Path(project_root).resolve()
    output = Path(output_dir)
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    if not output.is_relative_to(root) or output == root:
        raise DevelopmentCaseError("Alfentanil output must remain inside the project.")
    snapshot = _verify(root, *_ALFENTANIL_SOURCE, parse=True)
    _verify(root, *_ALFENTANIL_LITERATURE)
    identity = _verify(root, *_ALFENTANIL_IDENTITY, parse=True)
    if identity["CID"] != 51263 or identity["MolecularFormula"] != "C21H32N6O3":
        raise DevelopmentCaseError("Alfentanil identity does not support zero halogens.")
    definition = load_standard_human_model(root)
    source = _ALFENTANIL_SOURCE
    parameter_identification = _one(snapshot["ParameterIdentifications"], "Name", "global Parameter Identification")
    specs = []
    for identifier, old_name, expected, unit, path, search_scale, sensitivity_scale in (
        ("cyp3a4_intrinsic_clearance", "Intrinsic clearance", (.504, .1, 2.0), "l/min",
         "Alfentanil-CYP3A4-1st order CL|Intrinsic clearance", "log", "log"),
        ("lipophilicity", "Lipophilicity", (1.75, 1.3, 2.0), "Log Units",
         "Alfentanil|Lipophilicity", "linear", "bounded_linear"),
    ):
        original = _one(parameter_identification["IdentificationParameters"], "Name", old_name)
        values = [_one(original["Parameters"], "Name", name) for name in ("Start value", "MinValue", "MaxValue")]
        if tuple(item["Value"] for item in values) != expected or any(item["Unit"] != unit for item in values):
            raise DevelopmentCaseError("Alfentanil declared numerical starts/bounds changed.")
        if not any(link.endswith("|" + path) for link in original["LinkedParameters"]):
            raise DevelopmentCaseError("Historical parameter link does not establish the requested runtime path.")
        specs.append({"parameter_id": identifier, "initial_value": expected[0], "lower_bound": expected[1],
                      "upper_bound": expected[2], "unit": unit,
                      "runtime_mappings": [{"runtime_path": path, "multiplier": 1.0}],
                      "search_scale": search_scale, "sensitivity_scale": sensitivity_scale,
                      "bound_semantics": "historical_numerical_search_guards",
                      "source": _reference(*source, f"ParameterIdentifications[global Parameter Identification].IdentificationParameters[{old_name}]; Start value/MinValue/MaxValue, not optimized compound values")})
    compound = _plain(_one(snapshot["Compounds"], "Name", "Alfentanil"))
    # IV has no absorption task. Do not carry the fitted cellular permeability
    # or absorption alternatives from the former answer into the new base model.
    compound.pop("Permeability", None)
    compound.pop("IntestinalPermeability", None)
    compound["CalculationMethods"] = ["Cellular partition coefficient method - PK-Sim Standard", "Cellular permeability - PK-Sim Standard"]
    compound["Lipophilicity"][0].update(Name="Historical search start", IsDefault=True)
    compound["Lipophilicity"][0]["Parameters"][0]["Value"] = specs[1]["initial_value"]
    compound["FractionUnbound"][0]["Parameters"][0]["Value"] = .1
    solubility = _one(compound["Solubility"][0]["Parameters"], "Name", "Solubility at reference pH")
    if solubility["Value"] != 992 or solubility["Unit"] != "mg/l":
        raise DevelopmentCaseError("Alfentanil solubility source changed.")
    solubility.update(Value=.992, Unit="mg/ml")
    mw = _one(compound["Parameters"], "Name", "Molecular weight")
    compound["Parameters"] = [mw, *({"Name": atom, "Value": 0} for atom in ("F", "Cl", "Br", "I"))]
    metabolism = _one(compound["Processes"], "InternalName", "MetabolizationIntrinsic_FirstOrder")
    if metabolism["Molecule"] != "CYP3A4" or metabolism["DataSource"] != "1st order CL":
        raise DevelopmentCaseError("Unexpected Alfentanil metabolic process.")
    metabolism["Parameters"] = [{"Name": "Intrinsic clearance", "Value": specs[0]["initial_value"], "Unit": "l/min"}]
    filtration = _one(compound["Processes"], "InternalName", "GlomerularFiltration")
    if filtration["Parameters"] != [{"Name": "GFR fraction", "Value": .06}]:
        raise DevelopmentCaseError("Unexpected Alfentanil GFR fraction.")
    compound["Description"] = "Historical development split; clearance and logP are fresh search starts. Previous fitted permeability/clearance/logP removed."
    profile = _plain(_one([p for p in snapshot["ExpressionProfiles"] if p.get("Molecule") == "CYP3A4"],
                          "Category", "Standard European Male for DDI"))
    profile_id = "CYP3A4|Human|Standard European Male for DDI"
    if _one(profile["Parameters"], "Path", "CYP3A4|Reference concentration")["Value"] != 4.32:
        raise DevelopmentCaseError("Standard expression differs from its unfitted reference value.")
    compound_sources = {field: _reference(*source, f"Compounds[Alfentanil].{field}; development input")
                        for field in compound if field not in {"Name", "Description", "IsSmallMolecule"}}
    compound_sources.update(
        Parameters=_reference(*_ALFENTANIL_IDENTITY, "CID/formula-derived zero F/Cl/Br/I; MW416.52 from locked source snapshot"),
        Lipophilicity=copy.deepcopy(specs[1]["source"]),
        Processes=_reference(*source, "PI Intrinsic clearance start0.504; source GFR fraction0.06; no old optimum"),
        FractionUnbound=_reference(*_ALFENTANIL_LITERATURE, "Section2.2.1 fu10.0 percent; Edginton2008, not snapshot fit label"),
    )
    scenarios, external_scenarios, audit = [], [], []
    for study, role, simulation_name, count in _ALFENTANIL_STUDIES:
        simulation = _one(snapshot["Simulations"], "Name", simulation_name)
        individual = _plain(_one(snapshot["Individuals"], "Name", simulation["Individual"]))
        if individual["Name"] != "Standard European Male for DDI":
            raise DevelopmentCaseError("Do not inherit the fitted Kharasch2012 expression individual.")
        individual["OriginData"].pop("CalculationMethods", None)
        individual["ExpressionProfiles"] = [profile_id]
        protocol = _plain(_one(snapshot["Protocols"], "Name", simulation["Compounds"][0]["Protocol"]["Name"]))
        if protocol["ApplicationType"] != "IntravenousBolus" or protocol["DosingInterval"] != "Single":
            raise DevelopmentCaseError("Only locked single IV bolus scenarios are included.")
        observed = _one(snapshot["ObservedData"], "Name", simulation["ObservedData"][0])
        meta = _metadata(observed)
        if meta.get("Study Id") != study or meta.get("Species") != "Human" or meta.get("Molecule") != "Alfentanil" or meta.get("Route") != "IV":
            raise DevelopmentCaseError("Alfentanil study/route/source metadata mismatch.")
        scenario_id = "alfentanil_" + study.lower().replace(" ", "_") + "_iv"
        study_id = "alfentanil_" + study.lower().replace(" ", "_")
        original_protocol_name = protocol["Name"]
        protocol["Name"] = "PROTO_" + scenario_id
        points = _external_observations(observed, mw["Value"])
        if len(points) != count:
            raise DevelopmentCaseError("Alfentanil study point count changed.")
        for index, point in enumerate(points):
            point["observation_id"] = f"{scenario_id}:row_{index}"
        study_document = {"schema_version": "alfentanil-study-input-1.0", "scenario_id": scenario_id,
                          "compound_id": "pubchem:51263", "study_group_id": study_id, "source_study_id": study,
                          "role": role, "tier": "development", "metadata": meta, "unit": "umol/l", "time_unit": "min",
                          "source": _reference(*source, f"ObservedData[{observed['Name']}]"), "observations": points,
                          "source_governance": {"original_container_shared": True,
                                                "independence_unit": "publication_and_study_group_not_file_checksum",
                                                "all_studies_used_by_historical_public_model": True,
                                                "newly_blinded_independent_test": False}}
        path = output / ("external_observations" if role == "external" else "study_inputs") / (scenario_id + ".json")
        checksum = _write_once(path, study_document)
        reference = {"path": path.relative_to(root).as_posix(), "sha256": checksum}
        audit.append(reference)
        dose = _one(protocol["Parameters"], "Name", "InputDose")
        target = "Organism|PeripheralVenousBlood|Alfentanil|Plasma (Peripheral Venous Blood)"
        requirement = {"compound_semantic_id": "pubchem:51263", "name": "Alfentanil", "population": "healthy_adult",
                       "route": "intravenous", "dose": dose["Value"], "dose_unit": dose["Unit"], "formulation": None,
                       "targets": [{"target_id": "alfentanil_parent_plasma", "analyte": "parent", "matrix": "plasma",
                                    "quantity": "concentration", "binding": "total", "required": True}]}
        instance = {"population_context": "healthy_adult", "individual": individual,
                    "generated_demographics": [name for name in ("Weight", "Height") if name not in individual["OriginData"]],
                    "compound": copy.deepcopy(compound), "protocol": protocol, "snapshot_version": 80,
                    "simulation_name": scenario_id, "human_source": _reference(*source, "Individuals[Standard European Male for DDI]; male age30 is a historical model assumption, not each trial's measured mean"),
                    "compound_sources": copy.deepcopy(compound_sources),
                    "protocol_source": _reference(*source, f"Protocols[{original_protocol_name}]; new native-safe protocol identifier, unchanged dose"),
                    "formulation": None, "formulation_source": None,
                    "expression_profiles": [copy.deepcopy(profile)],
                    "expression_profile_sources": {profile_id: _reference(*source, "ExpressionProfiles[CYP3A4|Human|Standard European Male for DDI]; default/literature expression, not fitted Kharasch2012")},
                    "output_schema": _plain(simulation["OutputSchema"]), "output_selections": [target]}
        study_reference = {"study_id": study_id, "source_id": reference["path"], "source_checksum": checksum,
                           "role": role, "cohort_ids": [study_id], "observed_target_ids": ["alfentanil_parent_plasma"]}
        request = {"requirement": requirement, "human_model": {"registry_id": definition["registry_id"], "definition_checksum": definition["checksum"]},
                   "instance": instance, "base": None, "modules": [], "evidence": [],
                   "studies": [] if role == "external" else [study_reference]}
        scenario = {"scenario_id": scenario_id, "study_group_id": study_id, "role": role, "design_request": request,
                    "target_output": target, "unit": "umol/l", "time_unit": "min", "observation_count": count,
                    "source_study_id": study, "source_audit": {"source": reference, "original_container": source[0],
                                                               "original_container_sha256": source[1], "historical_model_building": True}}
        if role == "external":
            scenario["holdout_reference"] = reference
            external_scenarios.append(scenario)
        else:
            scenario["observations"] = points
            scenarios.append(scenario)
    case = {"case_id": "alfentanil_multistudy_iv_development", "compound_id": "pubchem:51263", "compound_name": "Alfentanil",
            "tier": "development", "purpose": "historical_development_reconstruction", "strict_main_eligible": False,
            "design_request": copy.deepcopy(scenarios[0]["design_request"]), "requirement": copy.deepcopy(scenarios[0]["design_request"]["requirement"]),
            "scenarios": scenarios, "forbidden_study_group_ids": [s["study_group_id"] for s in external_scenarios],
            "parameter_search": {"parameters": specs, "initial_values_are_optimal": False,
                                 "dataset_ids": [s["scenario_id"] for s in scenarios if s["role"] == "calibration"],
                                 "selection_dataset_ids": [s["scenario_id"] for s in scenarios if s["role"] == "selection"],
                                 "objective": "study_curve_equal_log_rmse",
                                 "aggregation": "sqrt(mean_study(mean_curve(mean_point(log_error_squared))))",
                                 "identifiability_verified": False, "runtime_units_must_be_verified": True},
            "study_counts": {"calibration": 2, "selection": 1, "test_loaded_by_designer": 0},
            "source_audit": [{"path": path, "sha256": sha} for path, sha in (_ALFENTANIL_SOURCE, _ALFENTANIL_LITERATURE, _ALFENTANIL_IDENTITY)] + audit[:3],
            "limitations": [
                "All selected studies were historical model-building data; this is a newly partitioned development exercise, not a newly blinded independent main benchmark.",
                "Publication/study groups are disjoint; formal cross-publication subject-level deduplication has not been established.",
                "Male age30 and default European physiology are historical model assumptions; study-specific age/weight/height were not available in the source individual.",
                "Intrinsic clearance0.504 and logP1.75 are historical optimization starts, not fitted final values or independent measurements.",
                "Search bounds are historical numerical guards, not biological confidence intervals.",
                "Fitted organ/intestinal permeability and the fitted Kharasch2012 CYP3A4 reference concentration are excluded; native cellular permeability is regenerated.",
                "External observations are stored separately and must remain unread by the model designer until predictions are locked.",
            ]}
    package = {"schema_version": "alfentanil-multistudy-development-1.0", "case": case,
               "external_scenarios": external_scenarios, "solver_calls": 0, "scientific_accepted": False}
    _write_once(output / "input_package.json", package)
    return package
