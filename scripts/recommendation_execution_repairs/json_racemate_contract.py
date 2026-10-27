"""Source-bound R/S omeprazole contracts for the shared native-project bridge.

No model is rebuilt here and no concentrations are summed outside the original
native observer. A metadata contract is not native execution or clinical validity.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

VERSION = "json-source-racemate-contract-1"
PARENTS = ("Esomeprazole", "R-omeprazole")
OBSERVER = "Omeprazole Plasma (Peripheral Venous Blood)"
OUTPUT_PATH = "Organism|PeripheralVenousBlood|Esomeprazole|" + OBSERVER
REGISTRY = "experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_identity_followup_20261002/racemate_source_registry.json"
REGISTRY_SHA256 = "acc7e64040e35bf74cbabdfafb89c0f1fa73b486f7054e86cd05fdc29886f7da"
MASS_MG = {"kg": Fraction(1000000), "g": Fraction(1000), "mg": Fraction(1), "µg": Fraction(1,1000), "ug": Fraction(1,1000)}
UNIT_CONVERSION_EVIDENCE = ({'path': 'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Domain/UnitSystem/Unit.cs', 'sha256': '25a3293d430aec77e7ec4a6d9b645c56ec82528fcc925280a19c26982a27ff8f'}, {'path': 'artifacts/pbpk-model-assets/gate6_discovery/snapshots/OSPSuite.Core/src/OSPSuite.Core/Domain/WithDisplayUnitExtensions.cs', 'sha256': '5ac28c97c36e4d1b4a5d26f0defc0c01da3f7b98d8f135f507d43fc922fcbb58'}, {'path': 'artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/PKSim.Core/Snapshots/Mappers/ParameterMapper.cs', 'sha256': '50c876b3ada8775221cba8b3829353ca9114c304257070cbda97b04386988c62'}, {'path': 'runtime/ospsuite-env/lib/R/library/ospsuite/lib/OSPSuite.Dimensions.xml', 'sha256': '69de334be1f2d35801cd44f90e26e63cae40974a8978189da527a0f62510322c'})
TIME_MIN = {"h": Fraction(60), "min": Fraction(1), "s": Fraction(1,60), "day(s)": Fraction(1440)}


def _require(ok, reason):
    if not ok: raise ValueError("DUAL_PARENT_" + reason)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _checked(ref, root):
    path = (root / ref["path"]).resolve(strict=True)
    _require(path.is_relative_to(root) and not any("sealed" in part.lower() for part in path.parts), "PROJECT_METADATA_PATH_REQUIRED")
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"], "EVIDENCE_CHANGED:" + ref["path"])
    return path


def _one(rows, name, label):
    found = [row for row in rows if row.get("Name") == name]
    _require(len(found) == 1, "UNIQUE_" + label + "_REQUIRED:" + name)
    return found[0]


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), "FINITE_NUMBER_REQUIRED")
    return Fraction(str(value))


def _semantic_schedule(value, *, parameter=False):
    # Names of protocols/schemas/items are identities, not timing or vehicle.
    # Parameter names remain semantic. The complete original object is also hashed.
    if isinstance(value, list):
        return [_semantic_schedule(v, parameter=isinstance(v, dict) and "Value" in v)
                for v in value if not (isinstance(v, dict) and v.get("Name") == "InputDose")]
    if isinstance(value, dict):
        return {k: _semantic_schedule(v) for k, v in value.items()
                if k not in {"Description", "ValueOrigin"} and (k != "Name" or parameter)}
    return value


def _protocol_channel(document, simulation, parent):
    compound = _one(simulation.get("Compounds", []), parent, "PARENT_SELECTION")
    binding = compound.get("Protocol")
    _require(isinstance(binding, dict) and bool(binding.get("Name")), "ORIGINAL_PARENT_PROTOCOL_REQUIRED:" + parent)
    protocol = _one(document.get("Protocols", []), binding["Name"], "PROTOCOL")
    item = protocol
    if "Schemas" in protocol:
        schemas = protocol["Schemas"]
        _require(len(schemas) == 1 and len(schemas[0].get("SchemaItems", [])) == 1, "PAIRED_SINGLE_ITEM_PROTOCOL_REQUIRED")
        item = schemas[0]["SchemaItems"][0]
    _require(item.get("ApplicationType") == "Oral", "ORIGINAL_ORAL_CHANNEL_REQUIRED")
    dose = _one(item.get("Parameters", []), "InputDose", "INPUT_DOSE")
    _require(dose.get("Unit") in MASS_MG, "ORIGINAL_ABSOLUTE_MASS_DOSE_REQUIRED")
    dose_mg = _number(dose.get("Value")) * MASS_MG[dose["Unit"]]
    _require(dose_mg > 0, "POSITIVE_ORIGINAL_CHANNEL_DOSE_REQUIRED")
    _require(not any(k in dose for k in ("Formula", "Table", "Distribution")), "EXPLICIT_ORIGINAL_DOSE_REQUIRED")
    for row in simulation.get("Parameters", []):
        parts = row.get("Path", "").split("|")
        if len(parts)>2 and parts[0] in {"Applications", "Events"} and parts[1] == binding["Name"]:
            _require(parts[-1] == "Volume of water/body weight", "UNPROVED_SOURCE_ADMINISTRATION_OVERRIDE")
    definition = _one(document.get("Compounds", []), parent, "COMPOUND")
    weight = _one(definition.get("Parameters", []), "Molecular weight", "MOLECULAR_WEIGHT")
    _require(weight.get("Unit") == "g/mol" and _number(weight.get("Value")) > 0, "SOURCE_MOLECULAR_WEIGHT_REQUIRED")
    return {"parent": parent, "protocol_name": binding["Name"], "protocol": deepcopy(protocol),
            "protocol_sha256": _digest(protocol), "selection": deepcopy(compound),
            "selection_sha256": _digest(compound), "compound_definition_sha256": _digest(definition),
            "source_dose_mg": float(dose_mg), "source_dose_rational_mg": [dose_mg.numerator, dose_mg.denominator],
            "molecular_weight_g_per_mol": weight["Value"], "formulations": deepcopy(binding.get("Formulations", [])),
            "schedule_signature_sha256": _digest(_semantic_schedule(protocol))}


def _source_contract(document, simulation_name):
    simulation = _one(document.get("Simulations", []), simulation_name, "SIMULATION")
    _require(simulation.get("Model") == "4Comp" and isinstance(simulation.get("Individual"), str)
             and not simulation.get("Population"), "ORIGINAL_INDIVIDUAL_4COMP_REQUIRED")
    individual = _one(document.get("Individuals", []), simulation["Individual"], "INDIVIDUAL")
    _require(individual.get("OriginData", {}).get("Species") == "Human", "ORIGINAL_HUMAN_REQUIRED")
    channels = [_protocol_channel(document, simulation, parent) for parent in PARENTS]
    _require(channels[0]["schedule_signature_sha256"] == channels[1]["schedule_signature_sha256"], "PAIRED_SCHEDULES_OR_VEHICLES_DIFFER")
    _require(channels[0]["formulations"] == channels[1]["formulations"], "PAIRED_FORMULATIONS_DIFFER")
    _require(channels[0]["molecular_weight_g_per_mol"] == channels[1]["molecular_weight_g_per_mol"], "PARENT_MOLECULAR_WEIGHTS_DIFFER")
    total = sum((Fraction(*c["source_dose_rational_mg"]) for c in channels), Fraction(0))
    for c in channels:
        fraction = Fraction(*c["source_dose_rational_mg"]) / total
        c["dose_fraction"] = {"numerator": fraction.numerator, "denominator": fraction.denominator}
    selected = [r.get("Name") for r in simulation.get("ObserverSets", [])]
    _require(selected.count(OBSERVER) == 1, "ORIGINAL_TOTAL_OBSERVER_NOT_SELECTED")
    observer_set = _one(document.get("ObserverSets", []), OBSERVER, "OBSERVER_SET")
    observer = _one(observer_set.get("Observers", []), OBSERVER, "OBSERVER")
    _require(observer.get("Dimension") == "Concentration (molar)" and observer.get("Type") == "Container"
        and observer.get("ContainerCriteria") == [{"Tag": "PeripheralVenousBlood", "Type": "MatchTag"}]
        and observer.get("MoleculeList") == {"ForAll": False, "MoleculeNamesToInclude": [PARENTS[0]]}, "ORIGINAL_TOTAL_OBSERVER_IDENTITY_DIFFERS")
    extras = sorted(c["Name"] for c in simulation["Compounds"] if c.get("Protocol") and c["Name"] not in PARENTS)
    return {"channels": channels, "observer_set": deepcopy(observer_set), "observer": deepcopy(observer),
            "other_administered_drugs": extras, "individual": simulation["Individual"],
            "simulation_definition_sha256": _digest({k:v for k,v in simulation.items()
                if k not in {"ObservedData", "OutputMappings", "IndividualAnalyses", "HasResults"}})}


def prepare_dual_parent_contract(provenance, simulation_name, clinical_contract, *, project_root):
    """Bind the frozen total-racemate design to both original native channels."""
    root = Path(project_root).resolve(strict=True)
    registry_ref = {"path": REGISTRY, "sha256": REGISTRY_SHA256}
    registry = json.loads(_checked(registry_ref, root).read_text())
    _require(_digest(clinical_contract) == registry["clinical_contract_sha256"], "EXACT_FROZEN_TOTAL_RACEMATE_CONTRACT_REQUIRED")
    row = registry["sources"].get(provenance.get("source_sha256"))
    _require(row is not None, "REGISTERED_ORIGINAL_SOURCE_REQUIRED")
    actual_source = (root / provenance["source_path"]).resolve(strict=True)
    _require(actual_source == _checked(row["source"], root), "ORIGINAL_SOURCE_PATH_DIFFERS")
    for key in ("author_document", "source_receipt", "verified_snapshot"):
        _checked(row[key], root)
    from .snapshot_repair import _verify_receipt
    receipt = _verify_receipt(root / row["source_receipt"]["path"], provenance, root)
    _require({k:receipt["snapshot"][k] for k in ("path", "sha256")} == row["verified_snapshot"], "RECEIPT_SNAPSHOT_DIFFERS")
    original = _source_contract(json.loads(actual_source.read_text()), simulation_name)
    verified = _source_contract(json.loads((root / row["verified_snapshot"]["path"]).read_text()), simulation_name)
    _require(_digest(original["observer_set"]) == row["observer_set_sha256"]
             and original["observer_set"] == verified["observer_set"], "REGISTERED_TOTAL_OBSERVER_CHANGED")
    for before, after in zip(original["channels"], verified["channels"]):
        _require(all(before[k] == after[k] for k in ("parent", "protocol_name", "dose_fraction", "source_dose_mg", "molecular_weight_g_per_mol", "formulations")), "SOURCE_CHANNEL_CHANGED_DURING_REPAIR")
        f = before["dose_fraction"]
        dose = _number(clinical_contract["dose_mg"]) * Fraction(f["numerator"], f["denominator"])
        before["target_dose_mg"] = float(dose)
        before["target_dose_rational_mg"] = [dose.numerator, dose.denominator]
        before["target_times_min"] = list(clinical_contract["administration_times_min"])
    _require(original["other_administered_drugs"] == verified["other_administered_drugs"], "OTHER_SOURCE_DRUGS_CHANGED")
    return {"version": VERSION, "registry": registry_ref, **deepcopy(row),
        "simulation_name": simulation_name, "clinical_contract_sha256": _digest(clinical_contract),
        **original, "output_path": OUTPUT_PATH,
        "molecular_weight_g_per_mol": original["channels"][0]["molecular_weight_g_per_mol"],
        "source_total_dose_mg": sum(c["source_dose_mg"] for c in original["channels"]),
        "target_total_dose_mg": clinical_contract["dose_mg"],
        "target_times_min": list(clinical_contract["administration_times_min"]),
        "analyte_mapping": "unchanged original native R+S molar observer; common original molecular weight converts the total once",
        "extra_drug_isolation_required": bool(original["other_administered_drugs"]),
        "native_execution_verified": False, "scientific_direct_validity_established": False}


def revalidate_contract(contract, clinical_contract, *, project_root):
    expected = prepare_dual_parent_contract({"source_path": contract["source"]["path"],
        "source_sha256": contract["source"]["sha256"]}, contract["simulation_name"], clinical_contract,
        project_root=project_root)
    _require(contract == expected, "CONTRACT_CHANGED")
    return expected


def validate_native_observer(document, dual_contract, *, molecular_weights, project_root):
    """Call only after the shared bridge binds this XML to source/simulation ID."""
    from .event_associations import ModelIndex
    from .source_pkml import element_sha
    root = Path(project_root).resolve(strict=True)
    registry = json.loads(_checked({"path": REGISTRY, "sha256": REGISTRY_SHA256}, root).read_text())
    revalidate_contract(dual_contract, registry["clinical_contract"], project_root=root)
    _require(dual_contract.get("version") == VERSION and dual_contract.get("output_path") == OUTPUT_PATH, "OUTPUT_CONTRACT_REQUIRED")
    _require(set(molecular_weights) == set(PARENTS) and all(
        type(molecular_weights[p]) in (int, float) and math.isclose(molecular_weights[p], dual_contract["molecular_weight_g_per_mol"], rel_tol=1e-12, abs_tol=1e-12) for p in PARENTS), "NATIVE_PARENT_MOLECULAR_WEIGHTS_DIFFER")
    _require(document.get("name") == dual_contract["simulation_name"] and bool(document.get("id")), "NATIVE_OBSERVER_SIMULATION_IDENTITY_REQUIRED")
    index = ModelIndex(document)
    weights = {}
    for parent in PARENTS:
        parameter = index.node(parent + "|Molecular weight", {"Para", "Parameter"}, "Molecular weight")
        _require(not parameter.get("rhs") and not parameter.get("rHSFormula"), "STATIC_MOLECULAR_WEIGHT_REQUIRED")
        raw = parameter.get("value")
        if parameter.get("formula"):
            formula = index.formulas.get(parameter.get("formula"))
            _require(formula is not None and formula.tag == "ConstantFormula" and len(formula) == 0, "CONSTANT_MOLECULAR_WEIGHT_REQUIRED")
            if raw is not None:
                _require(Fraction(raw) == Fraction(formula.get("value")), "CONFLICTING_NATIVE_MOLECULAR_WEIGHT")
            raw = formula.get("value")
        # PK-Sim's serialized base-unit double can differ in its last binary digits.
        # Bind the exact native node/hash and allow only conversion rounding,
        # using the same 1e-12 tolerance as independent complete-native readback.
        _require(raw is not None and math.isclose(float(Fraction(raw) * 1000000000),
            dual_contract["molecular_weight_g_per_mol"], rel_tol=1e-12, abs_tol=1e-12), "ORIGINAL_NATIVE_MOLECULAR_WEIGHT_DIFFERS")
        weights[parent] = {"path": parent + "|Molecular weight", "parameter_id": parameter.get("id"),
            "native_base_unit": "kg/µmol", "native_scalar": raw, "parameter_sha256": element_sha(parameter)}
    node = index.node(OUTPUT_PATH, {"Observer", "Obs"}, "Concentration (molar)")
    source_formula = dual_contract["observer"]["Formula"]
    refs = source_formula["References"]
    _require(len({r["Alias"] for r in refs}) == len(refs) == 15, "UNIQUE_ORIGINAL_OBSERVER_REFERENCES_REQUIRED")
    bindings = {r["Alias"]: r["Path"] for r in refs}
    for ref in refs:
        index.node(ref["Path"], dimension=ref["Dimension"])
    proof = index.formula(node, re.sub(r"\s+", "", source_formula["Formula"]), bindings)
    return {"verified": True, "output_path": OUTPUT_PATH, "output_entity_id": node.get("id"),
            "native_observer_sha256": element_sha(node), "native_formula": proof,
            "source_native_molecular_weights": weights,
            "original_observer_sha256": _digest(dual_contract["observer"]),
            "molecular_weight_g_per_mol": dual_contract["molecular_weight_g_per_mol"],
            "analyte": "total parent racemate", "dimension": "Concentration (molar)",
            "concentrations_aggregated_outside_native_model": False}


def _mass_unit_factors(project_root):
    """Use the registered runtime definition and official double arithmetic.

    PK-Sim ParameterMapper calls ConvertToBaseUnit; Unit performs
    (valueInUnit + Offset) * Factor. A decimal Fraction of the resulting
    binary64 string is not the original mathematical input dose.
    """
    root = Path(project_root).resolve(strict=True)
    for reference in UNIT_CONVERSION_EVIDENCE:
        _checked(reference, root)
    document = ET.parse(_checked(UNIT_CONVERSION_EVIDENCE[-1], root))
    dimensions = [n for n in document.iter('Dimension') if n.get('name') == 'Mass']
    _require(len(dimensions) == 1 and dimensions[0].get('baseUnit') == 'kg', 'OFFICIAL_MASS_BASE_UNIT_DIFFERS')
    factors = {}
    for unit in dimensions[0].findall('Units/Unit'):
        name = unit.get('name')
        _require(name not in factors, 'DUPLICATE_OFFICIAL_MASS_UNIT')
        factor, offset = float(unit.get('factor')), float(unit.get('offset', '0'))
        _require(math.isfinite(factor) and factor > 0 and offset == 0, 'OFFICIAL_MASS_FACTOR_OR_OFFSET_DIFFERS')
        factors[name] = factor
    _require(factors.get('kg') == 1.0 and factors.get('mg') == 1e-6, 'OFFICIAL_KG_MG_FACTORS_DIFFER')
    return factors


def validate_source_channel_dose(channel, source_dose, native_dose, *, project_root):
    """Exact source JSON -> official binary64 kg -> saved/native readback.

    No tolerance or accepted ULP interval is used. Even the adjacent binary64
    value is rejected, including if both saved and native values are altered.
    """
    protocol = channel['protocol']
    item = protocol
    if 'Schemas' in protocol:
        schemas = protocol['Schemas']
        _require(len(schemas) == 1 and len(schemas[0].get('SchemaItems', [])) == 1,
                 'PAIRED_SINGLE_ITEM_PROTOCOL_REQUIRED')
        item = schemas[0]['SchemaItems'][0]
    original = _one(item.get('Parameters', []), 'InputDose', 'INPUT_DOSE')
    value, unit = original.get('Value'), original.get('Unit')
    _require(unit in MASS_MG and _number(value) > 0, 'ORIGINAL_CHANNEL_MASS_REQUIRED')
    _require(_number(value) * MASS_MG[unit] == Fraction(*channel['source_dose_rational_mg']),
             'SOURCE_CHANNEL_CONTRACT_DOSE_DIFFERS')
    factors = _mass_unit_factors(project_root)
    _require(unit in factors and native_dose.get('unit') in factors, 'REGISTERED_NATIVE_MASS_UNIT_REQUIRED')
    _require(source_dose.get('dimension') == native_dose.get('dimension') == 'Mass',
             'SOURCE_PARAMETER_DIMENSIONS_DIFFER')
    _require(source_dose.get('formula_id') is None and source_dose.get('rhs_formula_id') is None,
             'SOURCE_DOSE_MUST_BE_EXPLICIT_STATIC_SCALAR')
    _number(native_dose.get('value'))
    expected_base = (float(value) + 0.0) * factors[unit]
    raw_saved = source_dose['value']
    _require(type(raw_saved) in (str, int, float), 'FINITE_SOURCE_DOSE_REQUIRED')
    saved_base = float(raw_saved)
    _number(saved_base)
    expected_native = expected_base / factors[native_dose['unit']] - 0.0
    _require(math.isfinite(saved_base) and saved_base.hex() == expected_base.hex(),
             'SAVED_SOURCE_DOSE_DIFFERS_FROM_OFFICIAL_CONVERSION')
    _require(float(native_dose['value']).hex() == expected_native.hex(), 'NATIVE_SOURCE_DOSE_DIFFERS')
    return {'original_value': value, 'original_unit': unit, 'source_base_unit': 'kg',
            'source_base_value_hex': saved_base.hex(), 'expected_base_value_hex': expected_base.hex(),
            'native_value_hex': float(native_dose['value']).hex(), 'native_unit': native_dose['unit'],
            'absolute_tolerance': 0, 'relative_tolerance': 0, 'allowed_ulp_difference': 0}


def build_dual_parent_patches(inventory, clinical_contract, source_metadata, dual_contract, *,
                             suppression_policies_by_parent=None, source_isolation=None, full_source_metadata=None, project_root):
    """Keep both parent channels active; use the unchanged complete patch checker."""
    from . import complete_candidate as complete
    from .coadministration_isolation import target_metadata
    revalidate_contract(dual_contract, clinical_contract, project_root=project_root)
    _require(_digest(clinical_contract) == dual_contract["clinical_contract_sha256"], "CLINICAL_CONTRACT_CHANGED")
    if dual_contract["other_administered_drugs"]:
        from .coadministration_isolation import verify_source_isolation
        _require(source_isolation is not None and full_source_metadata is not None, "EXTRA_DRUG_ISOLATION_REQUIRED")
        _require(source_isolation.get("target") == list(PARENTS)
            and source_isolation.get("zero_manifold_proof", {}).get("removed_administered_parents")
                == dual_contract["other_administered_drugs"], "EXACT_EXTRA_DRUG_AND_BOTH_RETAINED_PARENTS_REQUIRED")
        verify_source_isolation(source_isolation, full_source_metadata, clinical_contract, project_root=project_root)
        _require(target_metadata(full_source_metadata, list(PARENTS)) == source_metadata,
                 "RETAINED_NATIVE_METADATA_VIEW_DIFFERS")
    else:
        _require(source_isolation is None, "UNEXPECTED_EXTRA_DRUG_ISOLATION")
    _require(source_metadata.get("name") == dual_contract["simulation_name"], "NATIVE_SIMULATION_NAME_DIFFERS")
    source_groups, assignments = complete._metadata_events(source_metadata)
    _require(all(a.get("administered_compounds") in ([PARENTS[0]], [PARENTS[1]]) for a in assignments.values()), "EXTRA_OR_AMBIGUOUS_NATIVE_DRUG")
    _require({a["administered_compounds"][0] for a in assignments.values()} == set(PARENTS), "BOTH_NATIVE_PARENT_CHANNELS_REQUIRED")
    native_groups = {}
    for parameter in inventory.get("parameters", []):
        if not complete.native_pkml.ADMINISTRATION.fullmatch(parameter.get("path", "")): continue
        prefix, field = complete._event_prefix(parameter["path"])
        group = native_groups.setdefault(complete._normalized_event(prefix), {})
        _require(field not in group, "DUPLICATE_NATIVE_PARAMETER")
        group[field] = parameter
    _require(set(native_groups) == set(source_groups), "NATIVE_EVENT_COVERAGE_DIFFERS")
    for key, group in native_groups.items():
        _require(set(group) == set(source_groups[key]) and all(
            row["id"] == source_groups[key][field]["id"] for field, row in group.items()), "NATIVE_EVENT_FIELDS_OR_IDS_DIFFER")
    # The native source readback independently corroborates the original protocol ratio.
    native_schedules = []
    for channel in dual_contract["channels"]:
        parent = channel["parent"]
        events = []
        for key, assignment in assignments.items():
            if assignment["administered_compounds"] != [parent]: continue
            _require(key.startswith("Events|" + channel["protocol_name"] + "|"), "NATIVE_PROTOCOL_IDENTITY_DIFFERS")
            group = native_groups[key]; dose, start = group["Dose"], group["Start time"]
            _require(dose.get("unit") in MASS_MG and start.get("unit") in TIME_MIN, "NATIVE_CHANNEL_UNITS_REQUIRED")
            source_dose, source_start = source_groups[key]["Dose"], source_groups[key]["Start time"]
            _require(source_dose.get("dimension") == "Mass" and source_start.get("dimension") == "Time", "SOURCE_PARAMETER_DIMENSIONS_DIFFER")
            validate_source_channel_dose(channel, source_dose, dose, project_root=project_root)
            _require(Fraction(str(source_start["value"])) == _number(start["value"]) * TIME_MIN[start["unit"]],
                     "NATIVE_READBACK_DIFFERS_FROM_SAVED_SOURCE")
            events.append(_number(start["value"]) * TIME_MIN[start["unit"]])
        native_schedules.append(sorted(events))
    _require(native_schedules[0] == native_schedules[1] and bool(native_schedules[0]), "NATIVE_PAIRED_SCHEDULES_DIFFER")
    patches, event_rows = [], []
    for channel in dual_contract["channels"]:
        parent = channel["parent"]; metadata = target_metadata(source_metadata, parent)
        prefixes = set(complete._metadata_events(metadata)[0])
        selected = {"parameters": [r for r in inventory.get("parameters", [])
            if complete.native_pkml.ADMINISTRATION.fullmatch(r.get("path", ""))
            and complete._normalized_event(complete._event_prefix(r["path"])[0]) in prefixes]}
        target = deepcopy(clinical_contract); target["dose_mg"] = channel["target_dose_mg"]
        part, events = complete.build_patches(selected, target, metadata, native_compound_name=parent,
            suppression_policy=(suppression_policies_by_parent or {}).get(parent))
        patches.extend(part); event_rows.extend({**r, "parent": parent} for r in events)
    _require(len({p["path"] for p in patches}) == len(patches), "OVERLAPPING_PARENT_PATCHES")
    for i, time in enumerate(clinical_contract["administration_times_min"]):
        active = [e for e in event_rows if e["target_administration_index"] == i]
        _require(len(active) == 2 and {e["parent"] for e in active} == set(PARENTS)
            and all(e["start_time_min"] == time and e["dose_mg"] > 0 for e in active)
            and sum((_number(e["dose_mg"]) for e in active), Fraction(0)) == _number(clinical_contract["dose_mg"]), "TARGET_TOTAL_DOSE_NOT_PRESERVED")
    return patches, event_rows
