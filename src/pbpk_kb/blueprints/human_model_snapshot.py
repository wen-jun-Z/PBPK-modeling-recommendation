"""Pure, conservative PK-Sim 12 standard-human snapshot assembly.

Schema checked against the project's pksim_v12.3.173 Snapshot DTOs, CoreConstants,
ProjectVersions, ModelPropertiesTask and its PKSimDB standard-model definitions.
No file, database, network, model compiler or solver is opened by this module.
Only explicit caller values are copied; no example-compound values are inherited.
This deliberately supports a subset of PK-Sim, not arbitrary snapshot patching.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any
from uuid import UUID

STANDARD_MODEL_ID = "4Comp"
STANDARD_CONFIGURATION = "4Comp|Human"
SUPPORTED_SNAPSHOT_VERSION = 80
BUILDER_VERSION = "human-standard-snapshot-0.8.0-transport-development"
_POPULATIONS = {
    "Asian_Tanaka_1996", "BlackAmerican_NHANES_1997", "European_ICRP_2002",
    "Japanese_Population", "MexicanAmericanWhite_NHANES_1997", "WhiteAmerican_NHANES_1997",
}
_PARTITION_METHODS = {
    "Cellular partition coefficient method - PK-Sim Standard",
    "Cellular partition coefficient method - Rodgers and Rowland",
}
_PERMEABILITY_METHOD = "Cellular permeability - PK-Sim Standard"
_ENZYME_LOCALIZATIONS = {"Intracellular", "Intracellular, BloodCellsIntracellular, VascEndosome"}
_OTHER_PROTEIN_LOCALIZATIONS = {"Interstitial, BloodCellsMembrane, VascMembranePlasmaSide"}
# Native adult-human expression containers present in the archived source
# profiles. Only the enzyme's relative expression may be set in these organs.
_EXPRESSION_CONTAINERS = (
    "Bone", "Brain", "Fat", "Gonads", "Heart", "Kidney", "Liver|Pericentral", "Liver|Periportal", "Lung",
    "Muscle", "Pancreas", "Skin", "Spleen", "Stomach", "LargeIntestine",
    "LargeIntestine|Mucosa|Caecum", "LargeIntestine|Mucosa|ColonAscendens",
    "LargeIntestine|Mucosa|ColonTransversum", "LargeIntestine|Mucosa|ColonDescendens",
    "LargeIntestine|Mucosa|ColonSigmoid", "LargeIntestine|Mucosa|Rectum",
    "SmallIntestine", "SmallIntestine|Mucosa|Duodenum", "SmallIntestine|Mucosa|LowerIleum",
    "SmallIntestine|Mucosa|LowerJejunum", "SmallIntestine|Mucosa|UpperIleum",
    "SmallIntestine|Mucosa|UpperJejunum",
)
_PROCESSES = {
    "KidneyClearance": ("Renal", "Renal Clearances"),
    "GlomerularFiltration": ("GFR", "Glomerular Filtration"),
    "LiverClearance": ("Hepatic", "Total Hepatic Clearance"),
    "BiliaryClearance": ("Biliary", "Biliary Clearance"),
}
_ENZYME_PROCESS_PARAMETERS = {
    "MetabolizationIntrinsic_FirstOrder": {
        "Intrinsic clearance": {"l/min", "ml/min"}, "Specific clearance": {"1/min", "1/h"},
    },
    "rCYP450_FirstOrder": {
        "In vitro CL/recombinant enzyme": {"µl/min/pmol rec. enzyme"},
        "CLspec/[Enzyme]": {"l/µmol/min"},
    },
    "MetabolizationLiverMicrosomes_FirstOrder": {
        "In vitro CL for liver microsomes": {"µl/min/mg mic. protein"},
        "Content of CYP proteins in liver microsomes": {"pmol/mg mic. protein"},
        "CLspec/[Enzyme]": {"l/µmol/min"},
    },
    "MetabolizationSpecific_FirstOrder": {
        "Enzyme concentration": {"pmol/l", "nmol/l", "µmol/l", "µM", "mM", "mmol/l"},
        "Specific clearance": {"1/min", "1/h"}, "CLspec/[Enzyme]": {"l/µmol/min"},
    },
}
_CONCENTRATION_UNITS = {"pmol/l", "nmol/l", "µmol/l", "µM", "mM", "mmol/l"}
_MM_PROCESS_PARAMETERS = {
    "rCYP450_MM": {"In vitro Vmax/recombinant enzyme": {"nmol/min/pmol rec. enzyme", "pmol/min/pmol rec. enzyme"},
                   "Km": _CONCENTRATION_UNITS, "kcat": {"1/min"}},
    "MetabolizationLiverMicrosomes_MM": {
        "In vitro Vmax for liver microsomes": {"nmol/min/mg mic. protein", "pmol/min/mg mic. protein"},
        "Content of CYP proteins in liver microsomes": {"pmol/mg mic. protein"},
        "Km": _CONCENTRATION_UNITS, "kcat": {"1/min"}},
    "MetabolizationSpecific_MM": {"Enzyme concentration": _CONCENTRATION_UNITS,
        "Vmax": {"pmol/ml/min", "µmol/l/min"}, "Km": _CONCENTRATION_UNITS, "kcat": {"1/min"}},
    "MetabolizationIntrinsic_MM": {"Vmax (liver tissue)": {"pmol/min/g tissue", "µmol/min/kg tissue"},
        "Vmax": {"µmol/l/min"}, "Km": _CONCENTRATION_UNITS},
}
_INTERACTION_PROCESS_PARAMETERS = {
    "CompetitiveInhibition": {"Ki": _CONCENTRATION_UNITS},
    "NoncompetitiveInhibition": {"Ki": _CONCENTRATION_UNITS},
    "IrreversibleInhibition": {"kinact": {"1/h", "1/min"}, "K_kinact_half": _CONCENTRATION_UNITS,
                               "Ki": _CONCENTRATION_UNITS},
    "Induction": {"EC50": _CONCENTRATION_UNITS, "Emax": {None, ""}},
    "SpecificBinding": {"koff": {"1/min"}, "Kd": _CONCENTRATION_UNITS},
}
_TRANSPORT_PROCESS_PARAMETERS = {
    "ActiveTransportSpecific_MM": {"Transporter concentration": _CONCENTRATION_UNITS,
        "Vmax": {"µmol/l/min"}, "Km": _CONCENTRATION_UNITS, "kcat": {"1/min"}},
    "ActiveTransport_InVitro_VesicularAssay_MM": {
        "In vitro Vmax/transporter": {"nmol/min/pmol transporter", "pmol/min/pmol transporter"},
        "Km": _CONCENTRATION_UNITS, "kcat": {"1/min"}},
}
_MOLECULAR_PROCESS_PARAMETERS = {**_ENZYME_PROCESS_PARAMETERS, **_MM_PROCESS_PARAMETERS,
                                 **_INTERACTION_PROCESS_PARAMETERS, **_TRANSPORT_PROCESS_PARAMETERS}

# PK-Sim v12 TransportDirections and ExpressionContainer DTO. This is a
# native-location allowlist, not a new anatomical distribution or inferred
# transport direction. Each accepted profile supplies every explicit location.
_TRANSPORT_MUCOSA = {"Duodenum", "UpperJejunum", "LowerJejunum", "UpperIleum", "LowerIleum",
                     "Caecum", "ColonAscendens", "ColonTransversum", "ColonDescendens", "ColonSigmoid", "Rectum"}
_TRANSPORT_GENERAL = {"Bone", "Fat", "Gonads", "Heart", "Stomach", "SmallIntestine", "LargeIntestine",
                      "Lung", "Muscle", "Pancreas", "Skin", "Spleen"}
_TRANSPORT_DIRECTIONS = {
    "tissue": {"InfluxInterstitialToIntracellular", "EffluxIntracellularToInterstitial", "BiDirectionalInterstitialIntracellular", "PgpIntracellularToInterstitial"},
    "mucosa": {"InfluxLumenToMucosaIntracellular", "EffluxMucosaIntracellularToLumen", "BiDirectionalLumenMucosaIntracellular", "PgpMucosaIntracellularToLumen"},
    "brain_tissue": {"InfluxBrainInterstitialToTissue", "EffluxBrainTissueToInterstitial", "BiDirectionalBrainInterstitialTissue", "PgpBrainTissueToInterstitial"},
    "brain_plasma": {"InfluxBrainPlasmaToInterstitial", "EffluxBrainInterstitialToPlasma", "BiDirectionalBrainPlasmaInterstitial", "PgpBrainInterstitialToPlasma"},
    "BloodCells": {"InfluxPlasmaToBloodCells", "EffluxBloodCellsToPlasma", "BiDirectionalBloodCellsPlasma"},
    "VascularEndothelium": {"InfluxPlasmaToInterstitial", "EffluxInterstitialToPlasma", "BiDirectionalPlasmaInterstitial"},
}


def _validate_transport_expression(profile: dict) -> None:
    if profile.get("TransportType") not in {"Influx", "Efflux", "BiDirectional", "PgpLike"}:
        _fail("UNSUPPORTED_TRANSPORT_TYPE", "Transporter requires an explicit reviewed native type")
    locations = {(name, "Intracellular"): _TRANSPORT_DIRECTIONS["tissue"] for name in _TRANSPORT_GENERAL}
    for name in _TRANSPORT_MUCOSA | {"Kidney", "Pericentral", "Periportal"}:
        locations[(name, "Interstitial")] = _TRANSPORT_DIRECTIONS["tissue"]
        locations[(name, "Intracellular")] = (_TRANSPORT_DIRECTIONS["mucosa"] if name in _TRANSPORT_MUCOSA
            else {"ExcretionKidney"} if name == "Kidney" else {"ExcretionLiver"})
    locations.update({("Brain", "Plasma"): _TRANSPORT_DIRECTIONS["brain_plasma"],
                      ("Brain", "Interstitial"): _TRANSPORT_DIRECTIONS["brain_tissue"],
                      ("BloodCells", None): _TRANSPORT_DIRECTIONS["BloodCells"],
                      ("VascularEndothelium", None): _TRANSPORT_DIRECTIONS["VascularEndothelium"]})
    seen = set()
    for entry in _items(profile.get("Expression"), "transporter.Expression", minimum=1, maximum=len(locations)):
        _mapping(entry, {"Name", "CompartmentName", "TransportDirection"}, {"Name", "TransportDirection"}, "transporter.Expression")
        key = (entry["Name"], entry.get("CompartmentName"))
        if key in seen or key not in locations or entry["TransportDirection"] not in locations[key] | {"None"}:
            _fail("UNSUPPORTED_TRANSPORT_DIRECTION", "Source direction must match a unique registered organ/membrane location")
        seen.add(key)
    if seen != locations.keys():
        _fail("MISSING_TRANSPORT_DIRECTIONS", "Provide all explicit source membrane directions; do not infer missing locations")


def _enzyme_active_parameter(internal: str, names: set[str]) -> str | None:
    """Native first-order rate ultimately uses CLspec/[Enzyme]_app.

    Explicit downstream inputs take precedence over their upstream derivation;
    retain all source values, never invent zeros or convert process types.
    """
    if internal == "MetabolizationIntrinsic_FirstOrder":
        return "Specific clearance" if "Specific clearance" in names else "Intrinsic clearance" if "Intrinsic clearance" in names else None
    if "CLspec/[Enzyme]" in names:
        return "CLspec/[Enzyme]"
    expected = {"rCYP450_FirstOrder": "In vitro CL/recombinant enzyme",
                "MetabolizationLiverMicrosomes_FirstOrder": "In vitro CL for liver microsomes",
                "MetabolizationSpecific_FirstOrder": "Specific clearance"}.get(internal)
    return expected if expected in names else None


def _elimination_configured(process: dict) -> bool:
    parameters = {p["Name"]: p["Value"] for p in process["Parameters"]}
    if process["InternalName"] in _ENZYME_PROCESS_PARAMETERS:
        name = _enzyme_active_parameter(process["InternalName"], set(parameters))
    elif process["InternalName"] in _MM_PROCESS_PARAMETERS:
        capacities = ("kcat", "Vmax", "In vitro Vmax/recombinant enzyme", "In vitro Vmax for liver microsomes", "Vmax (liver tissue)")
        name = next((name for name in capacities if name in parameters), None)
    elif process["InternalName"] in _INTERACTION_PROCESS_PARAMETERS or process["InternalName"] in _TRANSPORT_PROCESS_PARAMETERS:
        # Uptake/redistribution alone does not establish whole-body elimination.
        return False
    elif process["InternalName"] == "GlomerularFiltration":
        name = "GFR fraction"
    else:
        name = "Specific clearance" if "Specific clearance" in parameters else "Plasma clearance"
    return parameters.get(name, 0) > 0
_ALTERNATIVE_GROUPS = {
    "Lipophilicity": "COMPOUND_LIPOPHILICITY",
    "FractionUnbound": "COMPOUND_FRACTION_UNBOUND",
    "Solubility": "COMPOUND_SOLUBILITY",
    "IntestinalPermeability": "COMPOUND_INTESTINAL_PERMEABILITY",
}
_PROVENANCE_KEYS = {
    "provenance_id", "provenance_kind", "source_id", "source_location", "source_checksum",
}


class HumanModelSnapshotError(ValueError):
    """Static input rejection; no executable artifact has been produced."""

    def __init__(self, code: str, detail: str, *, missing_fields: list[str] | None = None) -> None:
        self.code = code
        self.missing_fields = sorted(missing_fields or [])
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> None:
    raise HumanModelSnapshotError(code, detail)


def _mapping(value: Any, allowed: set[str], required: set[str], where: str) -> dict:
    if not isinstance(value, dict):
        _fail("INVALID_OBJECT", f"{where} must be a dictionary")
    if any(not isinstance(key, str) for key in value):
        _fail("INVALID_FIELD", f"{where} keys must be strings")
    extra = set(value) - allowed
    if extra:
        _fail("UNSUPPORTED_FIELD", f"{where}: unsupported fields {sorted(extra)}")
    missing = required - value.keys()
    if missing:
        fields = [f"{where}.{key}" for key in sorted(missing)]
        raise HumanModelSnapshotError("MISSING_FIELDS", ", ".join(fields), missing_fields=fields)
    return value


def _name(value: Any, where: str) -> str:
    if (not isinstance(value, str) or not value.strip() or value != value.strip()
            or any(character in value for character in "|/\\\n\r\t") or len(value) > 200):
        _fail("INVALID_IDENTIFIER", f"{where} must be a plain, nonempty PK-Sim name")
    return value


def _number(value: Any, where: str, minimum: float | None = None,
            maximum: float | None = None, *, positive: bool = False) -> float:
    try:
        finite = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        _fail("INVALID_NUMBER", f"{where} requires a finite real number")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        _fail("VALUE_OUT_OF_RANGE", where)
    if positive and value <= 0:
        _fail("VALUE_OUT_OF_RANGE", f"{where} must be positive")
    return float(value)


def _items(value: Any, where: str, *, minimum: int = 0, maximum: int = 64) -> list:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        _fail("INVALID_COLLECTION", f"{where} requires {minimum}..{maximum} items")
    return value


def _parameter(value: Any, where: str, units: set[str | None], *, named: bool = True) -> dict:
    allowed = {"Name", "Value", "Unit"} if named else {"Value", "Unit"}
    required = {"Name", "Value"} if named else {"Value"}
    item = _mapping(value, allowed, required, where)
    if named and (not isinstance(item["Name"], str) or not item["Name"].strip()):
        _fail("INVALID_IDENTIFIER", where + ".Name")
    _number(item["Value"], where + ".Value")
    if (item.get("Unit") is not None and not isinstance(item["Unit"], str)) or item.get("Unit") not in units:
        _fail("UNSUPPORTED_UNIT", f"{where}: unit {item.get('Unit')!r}")
    return item


def _parameters(values: Any, specifications: dict[str, set[str | None]],
                required: set[str], where: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for index, value in enumerate(_items(values, where)):
        if not isinstance(value, dict) or not isinstance(value.get("Name"), str) or value["Name"] not in specifications:
            _fail("UNSUPPORTED_PARAMETER", f"{where}[{index}]")
        name = value["Name"]
        if name in result:
            _fail("DUPLICATE_IDENTIFIER", f"{where}: {name}")
        result[name] = _parameter(value, f"{where}.{name}", specifications[name])
    missing = required - result.keys()
    if missing:
        fields = [f"{where}.{name}" for name in sorted(missing)]
        raise HumanModelSnapshotError("MISSING_FIELDS", ", ".join(fields), missing_fields=fields)
    return result


def _source(value: Any, where: str) -> dict:
    item = _mapping(value, _PROVENANCE_KEYS, _PROVENANCE_KEYS, where)
    try:
        UUID(str(item["provenance_id"]))
    except (ValueError, TypeError, AttributeError):
        _fail("INVALID_PROVENANCE", where + ".provenance_id")
    if not isinstance(item["provenance_kind"], str) or item["provenance_kind"] not in {
        "validated_evidence", "model_asset", "ontology_rule", "calibration_run", "user_request",
    }:
        _fail("INVALID_PROVENANCE", where + ".provenance_kind")
    for key in ("source_id", "source_location"):
        if not isinstance(item[key], str) or not item[key].strip():
            _fail("INVALID_PROVENANCE", where + "." + key)
    if not isinstance(item["source_checksum"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["source_checksum"]):
        _fail("INVALID_PROVENANCE", where + ".source_checksum")
    return item


def _validate_individual(individual: dict, generated_demographics: list[str], profile_ids: set[str]) -> None:
    allowed = {"Name", "OriginData", "Seed", "Description"}
    if profile_ids:
        allowed.add("ExpressionProfiles")
    _mapping(individual, allowed, {"Name", "OriginData", "Seed"}, "individual")
    _name(individual["Name"], "individual.Name")
    origin = _mapping(individual["OriginData"],
                      {"Species", "Population", "Gender", "Age", "Weight", "Height"},
                      {"Species", "Population", "Gender", "Age", "Weight", "Height"} - set(generated_demographics), "individual.OriginData")
    if origin["Species"] != "Human":
        _fail("NON_HUMAN_INDIVIDUAL", "OriginData.Species must be Human")
    if (not isinstance(origin["Population"], str) or origin["Population"] not in _POPULATIONS
            or not isinstance(origin["Gender"], str) or origin["Gender"] not in {"MALE", "FEMALE"}):
        _fail("UNSUPPORTED_PHYSIOLOGY", "Only listed standard nonpregnant human populations and sexes are supported")
    for key, unit in (("Age", "year(s)"), ("Weight", "kg"), ("Height", "cm")):
        if key in generated_demographics:
            if key in origin:
                _fail("AMBIGUOUS_DEMOGRAPHICS", f"{key} cannot be supplied and marked for native generation")
            continue
        parameter = _parameter(origin[key], "individual.OriginData." + key, {unit}, named=False)
        _number(parameter["Value"], key, positive=True)
    if type(individual["Seed"]) is not int or not 0 <= individual["Seed"] <= 2147483647:
        _fail("INVALID_SEED", "Seed must be an explicit nonnegative Int32")
    selected = individual.get("ExpressionProfiles", [])
    if not isinstance(selected, list) or any(not isinstance(x, str) for x in selected) or len(selected) != len(set(selected)) or set(selected) != profile_ids:
        _fail("EXPRESSION_PROFILE_BINDING_MISMATCH", "Individual must select exactly the supplied expression profiles")


def _validate_expression_profiles(profiles: list[dict], sources: dict[str, dict]) -> set[str]:
    """Sourced enzyme profiles with explicit core inputs; no anatomical edits."""
    identifiers, molecules = set(), set()
    for profile in _items(profiles, "expression_profiles", maximum=8):
        transporter = isinstance(profile, dict) and profile.get("Type") == "Transporter"
        keys = {"Type", "Species", "Molecule", "Category", "Localization", "Ontogeny", "Parameters"}
        if transporter:
            keys = (keys - {"Localization"}) | {"TransportType", "Expression"}
        _mapping(profile, keys, keys - {"Ontogeny"}, "expression_profile")
        molecule = _name(profile["Molecule"], "expression_profile.Molecule")
        locations = _ENZYME_LOCALIZATIONS if profile["Type"] == "Enzyme" else _OTHER_PROTEIN_LOCALIZATIONS
        if (profile["Type"] not in {"Enzyme", "OtherProtein", "Transporter"} or profile["Species"] != "Human"
                or not transporter and (not isinstance(profile["Localization"], str) or profile["Localization"] not in locations)):
            _fail("UNSUPPORTED_EXPRESSION_PROFILE", "An explicit human enzyme/protein profile with a reviewed native localization is required")
        if transporter:
            _validate_transport_expression(profile)
        _name(profile["Category"], "expression_profile.Category")
        if "Ontogeny" in profile and profile["Ontogeny"] != {"Name": molecule}:
            _fail("UNSUPPORTED_ONTOGENY", "The reviewed enzyme's ontogeny must be explicit")
        identifier = molecule + "|Human|" + profile["Category"]
        if identifier in identifiers or molecule in molecules:
            _fail("DUPLICATE_IDENTIFIER", "expression_profiles")
        identifiers.add(identifier)
        molecules.add(molecule)
        specs = {
            molecule + "|Reference concentration": {"µmol/l"},
            molecule + "|t1/2 (intestine)": {"h", "min"}, molecule + "|t1/2 (liver)": {"h", "min"},
        }
        required = set(specs)
        specs.update({f"Organism|{container}|Intracellular|{molecule}|Relative expression": {None, ""}
                      for container in _EXPRESSION_CONTAINERS})
        specs.update({molecule + "|" + name: {None, ""}
                      for name in ("Relative expression in blood cells", "Relative expression in plasma")})
        seen = set()
        for value in _items(profile["Parameters"], "expression_profile.Parameters", minimum=4, maximum=len(specs)):
            _mapping(value, {"Path", "Value", "Unit", "ValueOrigin"}, {"Path", "Value"}, "expression parameter")
            path = value["Path"]
            if not isinstance(path, str) or path not in specs or path in seen:
                _fail("UNSUPPORTED_EXPRESSION_PARAMETER", "Expression paths must be unique, registered molecular parameters")
            seen.add(path)
            _parameter({k: v for k, v in value.items() if k in {"Value", "Unit"}}, path, specs[path], named=False)
            _number(value["Value"], path, minimum=0, positive=path in required)
            if "ValueOrigin" in value:
                origin = _mapping(value["ValueOrigin"], {"Source", "Method", "Description"}, {"Source", "Description"}, path + ".ValueOrigin")
                if any(not isinstance(x, str) or not x.strip() for x in origin.values()):
                    _fail("INVALID_PROVENANCE", path + ".ValueOrigin")
        if not required <= seen or not seen - required:
            _fail("MISSING_FIELDS", "Expression profile requires explicit turnover, reference concentration and tissue expression inputs")
    _mapping(sources, identifiers, identifiers, "expression_profile_sources")
    for identifier, source in sources.items():
        _source(source, "expression_profile_sources." + identifier)
    return identifiers


def _alternative(compound: dict, key: str, specifications: dict[str, set[str | None]],
                 *, human: bool = False) -> dict[str, dict]:
    values = _items(compound[key], "compound." + key, minimum=1, maximum=1)
    item = _mapping(values[0], {"Name", "Parameters", "Species", "IsDefault"},
                    {"Name", "Parameters"} | ({"Species"} if human else set()), "compound." + key)
    _name(item["Name"], key + ".Name")
    if "Species" in item and item["Species"] != "Human":
        _fail("NON_HUMAN_PARAMETER", key)
    if "IsDefault" in item and item["IsDefault"] is not True:
        _fail("AMBIGUOUS_ALTERNATIVE", key + " must be the explicitly selected alternative")
    return _parameters(item["Parameters"], specifications, set(specifications), "compound." + key)


def _validate_compound(compound: dict, *, oral: bool, profile_ids: set[str], profile_types: dict[str, str]) -> list[dict]:
    required = {"Name", "IsSmallMolecule", "PlasmaProteinBindingPartner", "Parameters", "Lipophilicity",
                "FractionUnbound", "Solubility", "PkaTypes", "Processes", "CalculationMethods"}
    if oral:
        required.add("IntestinalPermeability")
    _mapping(compound, required | {"IntestinalPermeability", "Description"}, required, "compound")
    _name(compound["Name"], "compound.Name")
    if compound["IsSmallMolecule"] is not True:
        _fail("UNSUPPORTED_MOLECULE", "4Comp builder requires explicit IsSmallMolecule=true")
    if (not isinstance(compound["PlasmaProteinBindingPartner"], str)
            or compound["PlasmaProteinBindingPartner"] not in {"Albumin", "Glycoprotein", "Unknown"}):
        _fail("UNSUPPORTED_BINDING_PARTNER", "compound.PlasmaProteinBindingPartner")
    specs = {"Molecular weight": {"g/mol"}, **{name: {None, ""} for name in ("F", "Cl", "Br", "I")}}
    parameters = _parameters(compound["Parameters"], specs, set(specs), "compound.Parameters")
    _number(parameters["Molecular weight"]["Value"], "Molecular weight", positive=True)
    for name in ("F", "Cl", "Br", "I"):
        count = _number(parameters[name]["Value"], name, minimum=0)
        if not count.is_integer():
            _fail("INVALID_ATOM_COUNT", name)
    _alternative(compound, "Lipophilicity", {"Lipophilicity": {"Log Units"}})
    fraction = _alternative(compound, "FractionUnbound", {"Fraction unbound (plasma, reference value)": {None, ""}}, human=True)
    _number(fraction["Fraction unbound (plasma, reference value)"]["Value"], "FractionUnbound", maximum=1, positive=True)
    solubility = _alternative(compound, "Solubility", {"Solubility at reference pH": {"mg/ml"}, "Reference pH": {None, ""}})
    _number(solubility["Solubility at reference pH"]["Value"], "Solubility", positive=True)
    _number(solubility["Reference pH"]["Value"], "Reference pH", minimum=0, maximum=14)
    if "IntestinalPermeability" in compound:
        permeability = _alternative(compound, "IntestinalPermeability", {"Specific intestinal permeability (transcellular)": {"cm/s"}})
        _number(permeability["Specific intestinal permeability (transcellular)"]["Value"], "IntestinalPermeability", positive=True)
    seen_pka = set()
    for index, item in enumerate(_items(compound["PkaTypes"], "compound.PkaTypes", maximum=3)):
        _mapping(item, {"Type", "Pka"}, {"Type", "Pka"}, f"PkaTypes[{index}]")
        if not isinstance(item["Type"], str) or item["Type"] not in {"Acid", "Base"}:
            _fail("UNSUPPORTED_PKA_TYPE", "Use explicit empty PkaTypes for a neutral compound")
        _number(item["Pka"], "Pka")
        identity = (item["Type"], item["Pka"])
        if identity in seen_pka:
            _fail("DUPLICATE_IDENTIFIER", "PkaTypes")
        seen_pka.add(identity)
    methods = _items(compound["CalculationMethods"], "compound.CalculationMethods", minimum=2, maximum=2)
    if any(not isinstance(item, str) for item in methods) or len(set(methods)) != 2:
        _fail("INVALID_CALCULATION_METHODS", "Two distinct method IDs are required")
    if _PERMEABILITY_METHOD not in methods or len(set(methods) & _PARTITION_METHODS) != 1:
        _fail("UNSUPPORTED_CALCULATION_METHODS", "Provide one supported partition method and standard cellular permeability")
    selections = []
    used_types: set[str] = set()
    used_processes: set[tuple[str, str]] = set()
    for process in _items(compound["Processes"], "compound.Processes", minimum=1, maximum=16):
        enzyme = isinstance(process, dict) and process.get("InternalName") in _MOLECULAR_PROCESS_PARAMETERS
        required = {"InternalName", "DataSource", "Parameters"} | ({"Molecule"} if enzyme else {"Species"})
        _mapping(process, {"InternalName", "DataSource", "Species", "Parameters", "Description"}
                 | ({"Molecule", "Metabolite"} if enzyme else set()), required, "compound.Processes")
        internal = process["InternalName"]
        if enzyme:
            molecule = process.get("Molecule")
            if (not isinstance(molecule, str)
                    or not any(identifier.startswith(molecule + "|Human|") for identifier in profile_ids)):
                _fail("UNBOUND_ENZYME", "First-order metabolism requires its explicitly bound reviewed human enzyme profile")
            _name(process["DataSource"], "process.DataSource")
            if process.get("Species", "Human") != "Human":
                _fail("NON_HUMAN_PARAMETER", "process.Species")
            if process.get("Metabolite"):
                _fail("UNBOUND_METABOLITE", "The single-compound builder cannot invent or drop a named metabolite")
            expected_type = ("Transporter" if internal in _TRANSPORT_PROCESS_PARAMETERS else
                             "OtherProtein" if internal == "SpecificBinding" else "Enzyme")
            if profile_types.get(molecule) != expected_type:
                _fail("UNBOUND_ENZYME", "Native process requires the matching enzyme or binding-protein profile type")
            identity = (molecule, process["DataSource"])
            if identity in used_processes:
                _fail("DUPLICATE_IDENTIFIER", "enzyme process")
            used_processes.add(identity)
            specs = _MOLECULAR_PROCESS_PARAMETERS[internal]
            required = ({"Km"} if internal in _MM_PROCESS_PARAMETERS or internal in _TRANSPORT_PROCESS_PARAMETERS else
                        set(specs) - {"Ki"} if internal == "IrreversibleInhibition" else
                        set(specs) if internal in _INTERACTION_PROCESS_PARAMETERS else set())
            values = _parameters(process["Parameters"], specs, required, "molecular process")
            if internal in _ENZYME_PROCESS_PARAMETERS and _enzyme_active_parameter(internal, set(values)) is None:
                _fail("MISSING_FIELDS", "A sourced first-order clearance input is required")
            if (internal == "MetabolizationSpecific_FirstOrder" and "CLspec/[Enzyme]" not in values
                    and "Enzyme concentration" not in values):
                _fail("MISSING_FIELDS", "Specific first-order clearance needs source enzyme concentration")
            if internal in _MM_PROCESS_PARAMETERS and not set(values) & {
                "kcat", "Vmax", "In vitro Vmax/recombinant enzyme", "In vitro Vmax for liver microsomes", "Vmax (liver tissue)"}:
                _fail("MISSING_FIELDS", "Saturable kinetics requires an explicit source capacity as well as Km")
            if internal in _TRANSPORT_PROCESS_PARAMETERS:
                if not set(values) & {"kcat", "Vmax", "In vitro Vmax/transporter"}:
                    _fail("MISSING_FIELDS", "Active transport requires explicit source kcat or capacity with Km")
                if internal == "ActiveTransportSpecific_MM" and "kcat" not in values and "Transporter concentration" not in values:
                    _fail("MISSING_FIELDS", "Specific Vmax needs the source transporter concentration to derive kcat")
            for name, value in values.items():
                _number(value["Value"], name, minimum=0,
                        positive=name in {"Content of CYP proteins in liver microsomes", "Enzyme concentration", "Transporter concentration", "Km", "Ki", "Kd", "EC50", "K_kinact_half"})
            used_types.add(internal)
            selections.append({"Name": molecule + "-" + process["DataSource"], "MoleculeName": molecule})
            continue
        if not isinstance(internal, str) or internal not in _PROCESSES:
            _fail("UNSUPPORTED_PROCESS", "Only explicit systemic clearances and glomerular filtration are supported")
        if internal in used_types:
            _fail("DUPLICATE_IDENTIFIER", "Multiple alternatives for one systemic clearance are not supported")
        used_types.add(internal)
        _name(process["DataSource"], "process.DataSource")
        if process["Species"] != "Human":
            _fail("NON_HUMAN_PARAMETER", "process.Species")
        if internal == "GlomerularFiltration":
            # Native GFR uses the individual's renal physiology and compound fu;
            # its sole explicit compound input is the dimensionless fraction.
            specs = {"GFR fraction": {None, ""}}
            values = _parameters(process["Parameters"], specs, set(specs), "process." + internal)
            _number(values["GFR fraction"]["Value"], "process.GFR fraction", minimum=0, maximum=1)
        else:
            specs = {"Fraction unbound (experiment)": {None, ""},
                     "Plasma clearance": {"ml/min/kg", "l/h/kg", "ml/h/kg"},
                     "Specific clearance": {"1/min", "1/h"}}
            if internal != "KidneyClearance":
                specs["Lipophilicity (experiment)"] = {"Log Units"}
            required = set(specs) - {"Plasma clearance", "Specific clearance"}
            values = _parameters(process["Parameters"], specs, required, "process." + internal)
            _number(values["Fraction unbound (experiment)"]["Value"], "process.fu", maximum=1, positive=True)
            if not {"Plasma clearance", "Specific clearance"} & values.keys():
                _fail("MISSING_FIELDS", "A sourced plasma or specific clearance is required")
            for name in ("Plasma clearance", "Specific clearance"):
                if name in values:
                    _number(values[name]["Value"], "process." + name, minimum=0)
        process_type, display_name = _PROCESSES[internal]
        selections.append({"Name": display_name + "-" + process["DataSource"], "SystemicProcessType": process_type})
    if {"KidneyClearance", "GlomerularFiltration"} <= used_types:
        _fail("OVERLAPPING_CLEARANCE", "Systemic renal plasma clearance must not be combined with glomerular filtration")
    if {"LiverClearance", "BiliaryClearance"} <= used_types:
        _fail("OVERLAPPING_CLEARANCE", "Total hepatic and biliary clearances require an explicit nonoverlap derivation outside this builder")
    if "LiverClearance" in used_types and (set(_ENZYME_PROCESS_PARAMETERS) | set(_MM_PROCESS_PARAMETERS)) & used_types:
        _fail("OVERLAPPING_CLEARANCE", "Do not add systemic hepatic clearance to explicit hepatic metabolism")
    used_enzymes = {molecule for molecule, _ in used_processes}
    if {identifier.split('|', 1)[0] for identifier in profile_ids} != used_enzymes:
        _fail("UNUSED_EXPRESSION_PROFILE", "Every supplied protein profile must be used by an explicit selected molecular process")
    return selections


def _validate_protocol(protocol: dict, formulation: dict | None) -> dict:
    _mapping(protocol, {"Name", "ApplicationType", "DosingInterval", "Parameters", "Description"},
             {"Name", "ApplicationType", "DosingInterval", "Parameters"}, "protocol")
    _name(protocol["Name"], "protocol.Name")
    application = protocol["ApplicationType"]
    if not isinstance(application, str) or application not in {"IntravenousBolus", "Intravenous", "Oral"} or protocol["DosingInterval"] != "Single":
        _fail("UNSUPPORTED_ADMINISTRATION", "Only a single IV bolus, explicit IV infusion or oral dose is supported")
    specs = {"InputDose": {"mg", "mg/kg"}, "Start time": {"h"}}
    if application == "Oral":
        specs["Volume of water/body weight"] = {"ml/kg"}
    if application == "Intravenous":
        # ApplicationTypes.Intravenous is the native infusion application.
        # ProtocolMapper updates this dynamic parameter; PKSimDB declares Time/min.
        specs["Infusion time"] = {"min", "h"}
    values = _parameters(protocol["Parameters"], specs, set(specs), "protocol.Parameters")
    _number(values["InputDose"]["Value"], "dose", positive=True)
    _number(values["Start time"]["Value"], "Start time", minimum=0)
    if application == "Intravenous":
        _number(values["Infusion time"]["Value"], "Infusion time", positive=True)
    if application == "Oral":
        _number(values["Volume of water/body weight"]["Value"], "water volume", minimum=0)
        _mapping(formulation, {"Name", "FormulationType", "Parameters", "Description"},
                 {"Name", "FormulationType", "Parameters"}, "formulation")
        _name(formulation["Name"], "formulation.Name")
        if formulation["FormulationType"] == "Formulation_Dissolved":
            if formulation["Parameters"] != []:
                _fail("UNSUPPORTED_FORMULATION", "Dissolved formulation must explicitly carry no release parameters")
        elif formulation["FormulationType"] == "Formulation_Tablet_Weibull":
            if formulation["Parameters"] == []:
                _fail("UNSUPPORTED_FORMULATION", "A tablet cannot inherit an empty dissolved formulation contract")
            specifications = {"Dissolution time (50% dissolved)": {"min"}, "Lag time": {"min"},
                              "Dissolution shape": {None, ""}, "Use as suspension": {None, ""}}
            release = _parameters(formulation["Parameters"], specifications, set(specifications), "formulation.Parameters")
            for name in ("Dissolution time (50% dissolved)", "Dissolution shape"):
                _number(release[name]["Value"], name, positive=True)
            _number(release["Lag time"]["Value"], "Lag time", minimum=0)
            suspension = _number(release["Use as suspension"]["Value"], "Use as suspension", minimum=0, maximum=1)
            if suspension not in {0., 1.}:
                _fail("UNSUPPORTED_FORMULATION", "Native Use as suspension must be a source 0/1 switch")
        else:
            _fail("UNSUPPORTED_FORMULATION", "Only explicit dissolved or source-parameterized Weibull formulations are supported")
    elif formulation is not None:
        _fail("UNEXPECTED_FORMULATION", "IV administration must not carry an oral formulation")
    administration = {"route": "oral" if application == "Oral" else "intravenous",
            "dose": values["InputDose"]["Value"], "unit": values["InputDose"]["Unit"],
            "start_time_h": values["Start time"]["Value"], "application_type": application,
            "formulation_name": formulation["Name"] if formulation else None}
    if application == "Intravenous":
        duration = values["Infusion time"]
        administration["infusion_duration_min"] = duration["Value"] * (60 if duration["Unit"] == "h" else 1)
        _number(administration["infusion_duration_min"], "Infusion duration in minutes", positive=True)
    return administration


def _outputs(schema: list | None, selections: list[str] | None, compound_name: str) -> None:
    if schema is not None:
        previous_end = None
        for interval in _items(schema, "output_schema", minimum=1, maximum=32):
            _mapping(interval, {"Parameters"}, {"Parameters"}, "output_schema.interval")
            specs = {"Start time": {"h"}, "End time": {"h"}, "Resolution": {"pts/h"}}
            values = _parameters(interval["Parameters"], specs, set(specs), "output_schema.interval")
            start = _number(values["Start time"]["Value"], "output start", minimum=0)
            end = _number(values["End time"]["Value"], "output end", positive=True)
            _number(values["Resolution"]["Value"], "output resolution", positive=True)
            if end <= start or (previous_end is not None and start < previous_end):
                _fail("INVALID_OUTPUT_INTERVAL", "Output intervals must increase without overlap")
            previous_end = end
    if selections is not None:
        paths = _items(selections, "output_selections", minimum=1, maximum=64)
        if any(not isinstance(path, str) for path in paths) or len(set(paths)) != len(paths):
            _fail("DUPLICATE_OR_INVALID_OUTPUT", "Output paths must be unique strings")
        for path in paths:
            segments = path.split("|")
            if (len(segments) < 4 or segments[0] != "Organism" or segments.count(compound_name) != 1
                    or any(not item or item in {".", ".."} for item in segments)
                    or any(char in path for char in "/\\\n\r")):
                _fail("UNSUPPORTED_OUTPUT_PATH", "Output must be an Organism observer for the current compound")


def build_human_model_snapshot(
    *, individual: dict, compound: dict, protocol: dict, simulation_name: str,
    snapshot_version: int, human_source: dict, compound_sources: dict[str, dict],
    protocol_source: dict, formulation: dict | None = None, formulation_source: dict | None = None,
    output_schema: list | None = None, output_selections: list[str] | None = None,
    expected_compound_name: str | None = None,
    generated_demographics: list[str] | None = None,
    expression_profiles: list[dict] | None = None,
    expression_profile_sources: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Build a new standard-model input snapshot and a separate audit manifest.

    ``compound_sources`` maps each supplied compound field except Name,
    Description and IsSmallMolecule to a ProvenanceRef dictionary. References are
    structurally checked, never treated as proof that source values were read or
    scientifically validated. External evidence adapters must verify that link.
    Unsupported mechanisms/paths are rejected rather than silently dropped.
    """
    if type(snapshot_version) is not int or snapshot_version != SUPPORTED_SNAPSHOT_VERSION:
        _fail("UNSUPPORTED_SNAPSHOT_VERSION", "Only explicitly requested PK-Sim V12 snapshot Version=80 is supported")
    _name(simulation_name, "simulation_name")
    generated_demographics = [] if generated_demographics is None else generated_demographics
    _items(generated_demographics, "generated_demographics", maximum=2)
    if any(not isinstance(x, str) or x not in {"Weight", "Height"} for x in generated_demographics) or len(generated_demographics) != len(set(generated_demographics)):
        _fail("INVALID_DEMOGRAPHIC_POLICY", "Only absent Weight/Height may be explicitly delegated to PK-Sim generation")
    expression_profiles = [] if expression_profiles is None else expression_profiles
    expression_profile_sources = {} if expression_profile_sources is None else expression_profile_sources
    profile_ids = _validate_expression_profiles(expression_profiles, expression_profile_sources)
    _validate_individual(individual, generated_demographics, profile_ids)
    administration = _validate_protocol(protocol, formulation)
    selections = _validate_compound(compound, oral=administration["route"] == "oral", profile_ids=profile_ids,
                                    profile_types={profile["Molecule"]: profile["Type"] for profile in expression_profiles})
    if expected_compound_name is not None:
        _name(expected_compound_name, "expected_compound_name")
        if compound["Name"].casefold() != expected_compound_name.casefold():
            _fail("COMPOUND_IDENTITY_MISMATCH", "Compound name differs from the requested drug")
    names = [individual["Name"], compound["Name"], protocol["Name"], simulation_name]
    if formulation:
        names.append(formulation["Name"])
    if len(set(names)) != len(names):
        _fail("DUPLICATE_IDENTIFIER", "Use distinct instance building-block names")
    _outputs(output_schema, output_selections, compound["Name"])
    human_source = _source(human_source, "human_source")
    protocol_source = _source(protocol_source, "protocol_source")
    source_fields = set(compound) - {"Name", "Description", "IsSmallMolecule"}
    _mapping(compound_sources, source_fields, source_fields, "compound_sources")
    for field, reference in compound_sources.items():
        _source(reference, "compound_sources." + field)
    if formulation is not None:
        _source(formulation_source, "formulation_source")
    elif formulation_source is not None:
        _fail("UNEXPECTED_FORMULATION_SOURCE", "There is no formulation to bind")
    sources = [human_source, protocol_source, *compound_sources.values(), *expression_profile_sources.values()]
    if formulation_source is not None:
        sources.append(formulation_source)
    identities: dict[str, dict] = {}
    checksums: dict[str, str] = {}
    for source in sources:
        identity = str(source["provenance_id"])
        if identity in identities and identities[identity] != source:
            _fail("CONFLICTING_PROVENANCE", "One provenance identifier has different contents")
        if source["source_id"] in checksums and checksums[source["source_id"]] != source["source_checksum"]:
            _fail("CONFLICTING_PROVENANCE", "One source identifier has different checksums")
        identities[identity] = source
        checksums[source["source_id"]] = source["source_checksum"]
    # Bind supplied alternatives explicitly, including groups that also retain
    # provider-calculated alternatives after the native CompoundMapper runs.
    alternatives = [{"GroupName": group, "AlternativeName": compound[field][0]["Name"]}
                    for field, group in _ALTERNATIVE_GROUPS.items() if field in compound]
    # Native CompoundPropertiesMapper has no selection group for interaction
    # processes; SimulationMapper binds them separately by CompoundName.
    # Specific binding remains an ordinary compound process selection.
    compound_selections, interactions = [], []
    for process, item in zip(compound["Processes"], selections, strict=True):
        if process["InternalName"] in _INTERACTION_PROCESS_PARAMETERS and process["InternalName"] != "SpecificBinding":
            interactions.append({**item, "CompoundName": compound["Name"]})
        else:
            compound_selections.append(item)
    selection = {"Name": compound["Name"], "CalculationMethods": compound["CalculationMethods"],
                 "Alternatives": alternatives,
                 "Processes": compound_selections, "Protocol": {"Name": protocol["Name"]}}
    if formulation is not None:
        selection["Protocol"]["Formulations"] = [{"Key": "Formulation", "Name": formulation["Name"]}]
    simulation = {"Name": simulation_name, "Model": STANDARD_MODEL_ID,
                  "Individual": individual["Name"], "Compounds": [selection], "HasResults": False}
    if interactions:
        simulation["Interactions"] = interactions
    if output_schema is not None:
        simulation["OutputSchema"] = output_schema
    if output_selections is not None:
        simulation["OutputSelections"] = output_selections
    snapshot = {"Version": snapshot_version, "Individuals": [individual], "Compounds": [compound],
                "Protocols": [protocol], "Simulations": [simulation]}
    if formulation is not None:
        snapshot["Formulations"] = [formulation]
    if expression_profiles:
        snapshot["ExpressionProfiles"] = expression_profiles
    manifest = {
        "builder_version": BUILDER_VERSION, "artifact_kind": "pksim_build_input",
        "standard_model_id": STANDARD_MODEL_ID, "standard_model_name": "Standard model for small molecules",
        "configuration": STANDARD_CONFIGURATION, "generation_mode": "standard_model_instantiation",
        "snapshot_version": snapshot_version, "schema_reference": "PK-Sim 12.3.173 Snapshot DTOs",
        "model_properties_binding": "PKSim.ModelPropertiesTask.DefaultFor(OriginData, '4Comp')",
        "human_source": human_source, "individual_seed": individual["Seed"],
        "compound_sources": compound_sources,
        "protocol_source": protocol_source, "formulation_source": formulation_source,
        "expression_profile_sources": expression_profile_sources,
        "generated_demographics": {name: "to_be_generated_by_pksim_not_observed" for name in generated_demographics},
        "normalized_administration": administration,
        "clearance_mechanisms": sorted(set(
            "active_transport" if process["InternalName"] in _TRANSPORT_PROCESS_PARAMETERS
            else "renal" if item.get("SystemicProcessType") == "GFR"
            else item.get("SystemicProcessType", "Metabolism").lower()
            for process, item in zip(compound["Processes"], selections, strict=True)
            if process["InternalName"] not in _INTERACTION_PROCESS_PARAMETERS)),
        "elimination_configured": any(_elimination_configured(process) for process in compound["Processes"]),
        "source_binding_verified": False, "output_paths_runtime_verified": False,
        "source_clearance_parameterization": [{
            "internal_name": process["InternalName"], "molecule": process.get("Molecule"),
            "data_source": process["DataSource"], "supplied_parameters": [p["Name"] for p in process["Parameters"]],
            "explicit_downstream_input": _enzyme_active_parameter(process["InternalName"], {p["Name"] for p in process["Parameters"]})
                if process["InternalName"] in _ENZYME_PROCESS_PARAMETERS else None,
            "native_derivation_runtime_verified": False,
        } for process in compound["Processes"]],
        "compile_status": "not_run", "simulation_status": "not_run", "scientific_status": "not_evaluated",
        "execution": {"solver_calls": 0, "llm_calls": 0, "gpu_jobs": 0},
        "limitations": ["single_compound_single_dose", "standard_nonpregnant_human_physiology",
                        "reviewed_sourced_native_process_types_only", "oral_explicit_dissolved_or_source_weibull_formulation_only",
                        "runtime_mapping_and_mass_balance_not_verified"],
        "missing_execution_configuration": [name for name, value in
            (("output_schema", output_schema), ("output_selections", output_selections)) if value is None],
    }
    # Serialization checks all nested values and returns a fully detached object.
    try:
        payload = json.dumps({"snapshot": snapshot, "manifest": manifest}, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise HumanModelSnapshotError("INVALID_JSON_VALUE", "Inputs must contain only finite JSON values") from exc
    result = json.loads(payload)
    result["manifest"]["snapshot_checksum"] = hashlib.sha256(
        json.dumps(result["snapshot"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return result
