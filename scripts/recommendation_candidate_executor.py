"""Execute a frozen catalog candidate under an outcome-free clinical protocol.

This adapter reuses the complete registered compound model. It never fits a
parameter, constructs a replacement drug model, opens sealed observations, or
turns an execution failure into a scientific negative. Target-population and
formulation approximations remain explicit in the locked prediction.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pbpk_kb.blueprints.source_adaptation import (
    load_source_model, adapt_source_model, _administrations, _named, _parameters, _time,
    _event_override, _diff, _one_administered_selection, _check_dependencies,
)
from pbpk_kb.blueprints.source_workflow import read_curve
from pbpk_kb.blueprints.source_output_grid import ensure_output_times
from pbpk_kb.blueprints.source_adaptation_audit import (
    derive_administration_expectations, audit_execution,
)
from pbpk_kb.blueprints.design_runtime import execute_snapshot

VERSION = "catalog-candidate-native-executor-v1"
EXECUTION_REVISION = 'engineering_repair_02'
IMPLEMENTATION_FILES = (
    'scripts/recommendation_candidate_executor.py',
    'src/pbpk_kb/blueprints/source_adaptation_audit.py',
    'src/pbpk_kb/model_runtime/R/source_adaptation_audit.R',
    'src/pbpk_kb/blueprints/source_output_grid.py',
    'src/pbpk_kb/blueprints/source_adaptation.py',
    'src/pbpk_kb/blueprints/source_workflow.py',
    'src/pbpk_kb/blueprints/design_runtime.py',
    'src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R',
)
MASS_TO_MG = {"g": 1000., "mg": 1., "ug": .001, "µg": .001, "μg": .001, "ng": 1e-6}


class Blocked(ValueError):
    """A bounded execution prerequisite is absent; this is not scientific N."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _inside(value, root=ROOT, *, exists=True):
    root = Path(root).resolve(strict=True)
    path = (root / value).resolve(strict=exists)
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Path must remain inside this project")
    if any(part.casefold().startswith("sealed") for part in path.relative_to(root).parts):
        raise ValueError("Prediction adapter cannot open or write sealed resources")
    return path


def _sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def implementation_hashes():
    return {name: _sha(ROOT / name) for name in IMPLEMENTATION_FILES}


def _save(path, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if len(payload.encode()) > 16 * 1024**2:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if len(payload.encode()) > 16 * 1024**2:
        raise ValueError("Adapter JSON exceeds 16 MiB")
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        import os
        os.fsync(handle.fileno())


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Blocked("FINITE_NUMBER_REQUIRED:" + name)
    if value < 0 or (positive and value <= 0):
        raise Blocked("POSITIVE_NUMBER_REQUIRED:" + name)
    return float(value)


def _name(value):
    return " ".join(str(value).casefold().split())


def _form(value):
    text = _name(value)
    if "dissolved" in text:
        return "solution"
    for key in ("suspension", "solution", "capsule", "tablet"):
        if key in text:
            return key
    return None


def scenario_contract(scenario):
    """Normalize only design fields; never follow an observation reference."""
    if not isinstance(scenario, dict):
        raise Blocked("SCENARIO_MUST_BE_OBJECT")
    if any(key in scenario for key in ("observations", "observed_values", "ground_truth", "concentrations")):
        raise Blocked("OUTCOMES_FORBIDDEN_IN_PREDICTION_INPUT")
    raw = scenario.get("protocol_fields", scenario)
    if not isinstance(raw, dict):
        raise Blocked("PROTOCOL_FIELDS_MUST_BE_OBJECT")
    p = raw.get("protocol", {})
    n = scenario.get("normalized_design", {})
    for section in (raw, n):
        if any(key in section for key in ("observations", "observed_values", "ground_truth", "concentrations")):
            raise Blocked("OUTCOMES_FORBIDDEN_IN_PREDICTION_INPUT")
    dose = deepcopy(n.get("dose") or raw.get("dose") or {})
    if not isinstance(dose, dict):
        dose = {"value": dose}
    value = dose.get("value", p.get("dose"))
    unit = dose.get("unit", p.get("dose_unit"))
    if unit not in MASS_TO_MG:
        raise Blocked("ABSOLUTE_MASS_DOSE_REQUIRED:" + str(unit))
    labelled_mg = _number(value, "dose", positive=True) * MASS_TO_MG[unit]
    mg = labelled_mg
    active_mg = dose.get("active_moiety_dose_mg", p.get("active_moiety_dose_mg"))
    salt_conversion = p.get("salt_conversion", raw.get("salt_conversion"))
    if active_mg is not None:
        active_mg = _number(active_mg, "active-moiety dose", positive=True)
        if not isinstance(salt_conversion, dict):
            raise Blocked("ACTIVE_MOIETY_DOSE_REQUIRES_FROZEN_CONVERSION_SOURCE")
        factor = _number(salt_conversion.get("factor"), "salt conversion factor", positive=True)
        if not math.isclose(active_mg, labelled_mg*factor, rel_tol=1e-12, abs_tol=1e-12):
            raise Blocked("ACTIVE_MOIETY_CONVERSION_DOSE_CONFLICT")
        if not salt_conversion.get("source_file") or not salt_conversion.get("source_file_sha256"):
            raise Blocked("DOSE_CONVERSION_SOURCE_AND_CHECKSUM_REQUIRED")
        mg = active_mg
    elif dose.get("conversion_rule") and any(word in _name(dose.get("conversion_rule")) for word in ("salt", "model-species", "native", "hydration")):
        raise Blocked("SALT_TO_MODEL_SPECIES_DOSE_MAPPING_NOT_FROZEN")
    route = _name(n.get("route") or dose.get("route") or p.get("route"))
    if route in {"oral", "po", "p.o."}:
        route = "Oral"
    elif route in {"iv", "intravenous", "intravenous infusion", "intravenous_infusion", "iv_infusion"}:
        route = "Intravenous"
    elif route in {"intravenousbolus", "intravenous_bolus", "iv bolus", "iv_bolus"}:
        route = "IntravenousBolus"
    else:
        raise Blocked("UNSUPPORTED_TARGET_ROUTE:" + route)
    regimen = n.get("regimen") or {}
    origin = regimen.get("observation_origin_h", dose.get("observation_origin_h", 0))
    if origin is None:
        raise Blocked("UNKNOWN_OBSERVATION_ORIGIN_REQUIRES_EXPLICIT_PROTOCOL")
    origin = _number(origin, "observation origin")
    if regimen.get('coadministered_drugs') or dose.get('coadministered_drugs'):
        raise Blocked('COADMINISTERED_TARGET_DRUGS_REQUIRE_SEPARATE_BOUND_ADAPTER')
    administration = regimen.get("administrations", p.get("administrations"))
    if administration is not None:
        if not isinstance(administration, list) or not 1 <= len(administration) <= 30:
            raise Blocked("EXPLICIT_ONE_TO_THIRTY_ADMINISTRATIONS_REQUIRED")
        hours = []
        for row in administration:
            if not isinstance(row, dict):
                raise Blocked("EXPLICIT_ADMINISTRATION_OBJECT_REQUIRED")
            time = row.get("time_min")
            if time is None and row.get("time_h") is not None:
                time = _number(row["time_h"], "administration time") * 60
            hours.append(_number(time, "administration time") / 60)
            if row.get("dose") is not None:
                au = row.get("dose_unit", unit)
                if au not in MASS_TO_MG or not math.isclose(_number(row["dose"], "administration dose") * MASS_TO_MG[au], labelled_mg):
                    raise Blocked("CONFLICTING_TARGET_DOSES")
    else:
        hours = regimen.get("dose_times_h", regimen.get("administrations_h", dose.get("administrations_h")))
        if hours is None:
            hours = regimen.get("administration_times_h")
        if not isinstance(hours, list) or not 1 <= len(hours) <= 30:
            raise Blocked("EXPLICIT_ONE_TO_THIRTY_ADMINISTRATIONS_REQUIRED")
        hours = [_number(t, 'administration time') for t in hours]
    if hours[0] != 0:
        raise Blocked('TARGET_ADMINISTRATION_MUST_START_AT_ZERO')
    if any(b <= a for a, b in zip(hours, hours[1:])):
        raise Blocked('TARGET_ADMINISTRATION_TIMES_MUST_STRICTLY_INCREASE')
    if len(hours) > 1 and any(not math.isclose(t, i * hours[1], rel_tol=0, abs_tol=1e-9) for i, t in enumerate(hours)):
        raise Blocked('NONREGULAR_TARGET_REGIMEN_REQUIRES_SEPARATE_REGISTERED_ADAPTER')
    if not any(math.isclose(origin, t, rel_tol=0, abs_tol=1e-9) for t in hours):
        raise Blocked('OBSERVATION_ORIGIN_NOT_A_DECLARED_ADMINISTRATION')
    infusion = dose.get("infusion_duration_min", regimen.get("infusion_duration_min",
        n.get("infusion_duration_min", p.get("infusion_duration_min"))))
    if infusion is None and regimen.get("infusion_duration_h") is not None:
        infusion = _number(regimen["infusion_duration_h"], "infusion duration") * 60
    if route == "Intravenous":
        infusion = _number(infusion, "infusion duration", positive=True)
    else:
        infusion = None
    sampling = n.get("sampling") or {}
    times_h = sampling.get("times_h", raw.get("nominal_sampling_hours_from_observation_origin"))
    if times_h is None and p.get("sampling_times_min") is not None:
        minute_grid = p["sampling_times_min"]
        if not isinstance(minute_grid, list) or any(isinstance(x, bool) or not isinstance(x, (int, float))
                or not math.isfinite(x) for x in minute_grid):
            raise Blocked("FINITE_NUMBER_REQUIRED:sampling time")
        times_h = [x / 60 for x in minute_grid]
    if not isinstance(times_h, list) or len(times_h) < 2 or len(times_h) > 2048:
        raise Blocked("EXPLICIT_BOUNDED_SAMPLING_TIMES_REQUIRED")
    for x in times_h:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise Blocked("FINITE_NUMBER_REQUIRED:sampling time")
    predose = [x * 60 for x in times_h if x < 0]
    times = [float(x) * 60 for x in times_h if x >= 0]
    if len(times) < 2:
        raise Blocked("AT_LEAST_TWO_POSTDOSE_SAMPLING_TIMES_REQUIRED")
    if any(b <= a for a, b in zip(times, times[1:])) or times[-1] <= 0:
        raise Blocked("SAMPLING_TIMES_MUST_INCREASE")
    target = raw.get("evaluation_target", {})
    matrix = _name(n.get("matrix") or raw.get("matrix") or target.get("matrix"))
    if matrix not in {"plasma", "whole_blood"}:
        raise Blocked("PARENT_TOTAL_PLASMA_OR_WHOLE_BLOOD_OUTPUT_ONLY:" + matrix)
    analyte = _name(n.get("analyte") or raw.get("analyte") or target.get("analyte"))
    positive_analyte = analyte.replace("not metabolite or unbound concentration", "")
    if "parent" not in positive_analyte or "unbound" in positive_analyte or "metabolite" in positive_analyte:
        raise Blocked("EXPLICIT_PARENT_ANALYTE_REQUIRED")
    if _name(target.get("binding", "total")) != "total":
        raise Blocked("TOTAL_PARENT_OUTPUT_ONLY")
    return {"case_id": scenario.get("case_id", scenario.get("scenario_id", raw.get("case_id"))),
        "compound": _name(scenario.get("compound", raw.get("compound"))), "route": route,
        "dose_mg": mg, "reported_labelled_dose_mg": labelled_mg,
        "salt_conversion": deepcopy(salt_conversion),
        "dose_basis": dose.get("basis", p.get("dose_basis")),
        "infusion_minutes": infusion, "sampling_times_min": times,
        "administration_times_min": [t * 60 for t in hours],
        "observation_origin_min": origin * 60,
        "absolute_sampling_times_min": [origin * 60 + t for t in times],
        "regimen_source": "explicit_frozen_clinical_times; no inferred steady state or dosing history",
        "predose_sampling_times_min_not_simulated": predose,
        "formulation": dose.get("formulation", p.get("formulation", n.get("formulation"))),
        "population": deepcopy(n.get("population", raw.get("population", p))),
        "food": deepcopy(n.get("food", raw.get("food", p.get("food_state")))),
        "prediction_contract": deepcopy(n.get("prediction_contract", raw.get("prediction_contract", {}))),
        "native_binding": deepcopy(raw.get("native_binding", {})),
        "matrix": matrix, "analyte": "parent", "binding": "total"}


def _source_options(snapshot, contract, *, source_sha256=None):
    options, rejected = [], []
    for sim in snapshot.get("Simulations", []):
        name = sim.get("Name", "")
        try:
            if sim.get("Model") != "4Comp" or sim.get("Population"):
                raise Blocked("INDIVIDUAL_4COMP_REQUIRED")
            selection = _one_administered_selection(sim)
            if _name(selection.get("Name")) != contract["compound"]:
                raise Blocked("EXACT_SINGLE_PARENT_COMPOUND_REQUIRED")
            _check_dependencies({**snapshot, 'Simulations': [sim]})
            individual = _named(snapshot.get("Individuals"), sim.get("Individual"), "Individuals")
            if individual.get("OriginData", {}).get("Species") != "Human":
                raise Blocked("HUMAN_SOURCE_REQUIRED")
            protocol = _named(snapshot.get("Protocols"), selection.get("Protocol", {}).get("Name"), "Protocols")
            items, times = _administrations(protocol)
            if not times:
                raise Blocked("SOURCE_REQUIRES_UNAMBIGUOUS_ADMINISTRATION_TIMES")
            from pbpk_kb.blueprints.source_adaptation import _uniform_vehicle_items
            item = _uniform_vehicle_items(items)
            if item.get("ApplicationType") != contract["route"]:
                raise Blocked("SOURCE_ROUTE_DIFFERS")
            d = _parameters(item).get("InputDose", {})
            weighted = isinstance(d.get('Unit'), str) and d['Unit'].endswith('/kg')
            base_unit = d['Unit'][:-3] if weighted else d.get('Unit')
            if base_unit not in MASS_TO_MG:
                raise Blocked("SOURCE_MASS_OR_MASS_PER_KG_DOSE_REQUIRED")
            source_dose_value = _number(d.get('Value'), 'source dose', positive=True)
            # Do not estimate native weight from age/sex or substitute a target
            # mass as the source mass. Unknown mass simply loses dose-distance
            # tie breaking; lexical source identity keeps ordering deterministic.
            smg = None if weighted else source_dose_value * MASS_TO_MG[base_unit]
            matrix = contract.get("matrix", "plasma")
            if matrix == "whole_blood":
                # Use the source's own blood observer, including its red-cell
                # binding/distribution mechanisms. No plasma-to-blood factor.
                eligible_outputs = {
                    "Organism|PeripheralVenousBlood|" + selection["Name"] + "|Whole Blood (Peripheral Venous Blood)"}
            elif matrix == "plasma":
                eligible_outputs = {
                    "Organism|PeripheralVenousBlood|" + selection["Name"] + "|Plasma (Peripheral Venous Blood)",
                    "Organism|VenousBlood|Plasma|" + selection["Name"] + "|Concentration in container"}
            else:
                raise Blocked("SOURCE_OUTPUT_MATRIX_UNSUPPORTED:" + str(matrix))
            outputs = [x for x in sim.get("OutputSelections", []) if x in eligible_outputs]
            if not outputs:
                raise Blocked("EXPLICIT_TOTAL_PARENT_" + matrix.upper() + "_OUTPUT_REQUIRED")
            output = sorted(outputs, key=lambda x: ("PeripheralVenousBlood" not in x, x))[0]
            target_form = _form(contract.get("formulation"))
            source_forms = selection.get("Protocol", {}).get("Formulations", [])
            source_form = None
            if len(source_forms) == 1:
                named_form = _named(snapshot.get("Formulations"), source_forms[0]["Name"], "Formulations")
                source_form = _form(named_form.get("Name")) or _form(named_form.get("FormulationType"))
            binding = contract.get("native_binding", {})
            bound_form = (binding.get("formulation_name")
                if binding.get("source_snapshot", {}).get("sha256") == source_sha256 and source_sha256 is not None else None)
            # A later exact-source-bound formulation switch is checked by the
            # source adapter. Without such a mapping, do not substitute a
            # dissolved source for a requested solid formulation.
            if (contract["route"] == "Oral" and not bound_form and target_form and source_form
                    and target_form != source_form):
                raise Blocked("SOURCE_FORMULATION_CLASS_DIFFERS")
            # Selection uses design and source metadata only, never performance.
            food = _name(contract.get("food"))
            # Word boundaries are essential: breakfast is not a fasted state.
            fast = bool(re.search(r'\b(?:fasted|fasting|fast)\b', food) or '\u7a7a\u8179' in food
                        or (isinstance(contract.get('food'), dict) and any('fasting_h' in k for k in contract['food'])))
            fed = bool(re.search(r'\bfed\b|\bhigh[- ]fat\b|\bwith (?:a )?(?:meal|food|breakfast)\b', food))
            source_fast = bool(re.search(r'\b(?:fasted|fasting|fast)\b', _name(name)))
            source_fed = bool(re.search(r'\bfed\b|\bwith (?:a )?(?:meal|food|breakfast)\b', _name(name)))
            food_penalty = 0 if ((fast and source_fast) or (fed and source_fed)) else 1
            if fast and not fed and source_fed:
                raise Blocked("EXPLICIT_SOURCE_FED_TARGET_FASTED_CONFLICT")
            if fed and not fast and source_fast:
                raise Blocked('EXPLICIT_SOURCE_FASTED_TARGET_FED_CONFLICT')
            options.append({"name": name, "compound_name": selection["Name"], "output_path": output,
                "individual": deepcopy(individual), "source_dose_mg": smg,
                "source_formulation_class": source_form,
                "source_reported_dose": deepcopy(d),
                "source_dose_basis": 'mass_per_kg_absolute_mass_not_inferred' if weighted else 'absolute_mass',
                "sort_key": (food_penalty, abs(math.log(smg / contract["dose_mg"])) if smg is not None else math.inf, name)})
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"simulation": name, "reason": str(exc)})
    return sorted(options, key=lambda row: row["sort_key"]), rejected


def apply_explicit_regimen(snapshot, contract):
    """Bind frozen regular clinical times without changing drug/physiology values.

    Only one selected administered parent, one source administration item and
    equal target doses are supported. Source event vehicle overrides must be
    consistent across source administrations before being reused. Every target
    event is independently checked against the protocol by native readback.
    """
    result = deepcopy(snapshot)
    sim = result['Simulations'][0]
    _check_dependencies(result)
    reference = _one_administered_selection(sim)['Protocol']
    old_name = reference['Name']
    old_protocol = _named(result['Protocols'], old_name, 'Protocols')
    items, old_times = _administrations(old_protocol)
    from pbpk_kb.blueprints.source_adaptation import _uniform_vehicle_items
    _uniform_vehicle_items(items)
    target = contract['administration_times_min']
    effective_times = list(old_times)
    for parameter in sim.get('Parameters', []):
        override = _event_override(parameter, old_name)
        if override and override['parameter'] == 'Start time':
            if override['application_index'] > len(effective_times):
                raise Blocked('SOURCE_OVERRIDE_REFERENCES_NONEXISTENT_EVENT')
            effective_times[override['application_index'] - 1] = _time(parameter)
    if len(items) == 1 and old_times == target and effective_times == target:
        return {'snapshot': result, 'manifest': {'changed': False, 'explicit_times_min': target,
                'observation_origin_min': contract['observation_origin_min'], 'drug_model_modified': False}, 'diff': []}
    item = deepcopy(items[0])
    parameters = _parameters(item)
    if 'Start time' not in parameters:
        raise Blocked('SOURCE_ITEM_START_TIME_REQUIRED')
    parameters['Start time'].update(Value=0., Unit='min')
    parameters['Start time'].pop('ValueOrigin', None)
    item.pop('DosingInterval', None)
    item.pop('Name', None)
    item.pop('Description', None)
    if contract['route'] == 'Oral':
        # A simple source omits FormulationKey; advanced native schemas require it.
        if len(reference.get('Formulations', [])) != 1:
            raise Blocked('REPEATED_REGIMEN_REQUIRES_ONE_SOURCE_FORMULATION')
        item['FormulationKey'] = reference['Formulations'][0]['Key']
    new_name = 'clinical_schedule_' + _digest({'source_protocol': old_name, 'times': target,
                                               'dose_mg': contract['dose_mg']})[:20]
    if any(p.get('Name') == new_name for p in result['Protocols']):
        raise Blocked('GENERATED_REGIMEN_NAME_COLLISION')
    protocol = {'Name': new_name, 'Schemas': [{'SchemaItems': [item], 'Parameters': [
        {'Name': 'Start time', 'Value': 0., 'Unit': 'min'},
        {'Name': 'NumberOfRepetitions', 'Value': len(target)},
        {'Name': 'TimeBetweenRepetitions', 'Value': target[1] if len(target) > 1 else 0., 'Unit': 'min'}]}]}
    old_override_templates, kept = {}, []
    for parameter in sim.get('Parameters', []):
        override = _event_override(parameter, old_name)
        if override is None:
            # Unrelated existing event overrides would introduce unbound dosing.
            if parameter.get('Path', '').split('|')[0] in {'Events', 'Applications'}:
                raise Blocked('UNRELATED_EVENT_OVERRIDE_REQUIRES_SEPARATE_BINDING')
            kept.append(parameter)
            continue
        if override['application_index'] > len(old_times):
            raise Blocked('SOURCE_OVERRIDE_REFERENCES_NONEXISTENT_EVENT')
        parts = override['parts']
        app = next(i for i, text in enumerate(parts) if re.fullmatch(r'Application_[1-9][0-9]*', text))
        parts[0], parts[1], parts[app] = 'Events', new_name, 'Application_{index}'
        field = override['parameter']
        if field in {'InputDose', 'Dose', 'Start time', 'Infusion time'}:
            # Dose/time/duration are specified by the new protocol. Preserve the
            # exact field and native path, replacing only its requested value.
            canonical = None
        else:
            canonical = {k: parameter.get(k) for k in ('Value', 'Unit', 'TableFormula')}
        key = '|'.join(parts)
        if key in old_override_templates and old_override_templates[key]['canonical'] != canonical:
            raise Blocked('NONUNIFORM_SOURCE_EVENT_OVERRIDE_CANNOT_BE_REPLICATED:' + field)
        old_override_templates[key] = {'parameter': deepcopy(parameter), 'field': field, 'canonical': canonical}
    for key, template in old_override_templates.items():
        for index, time in enumerate(target, 1):
            row = deepcopy(template['parameter'])
            row['Path'] = key.format(index=index)
            field = template['field']
            if field in {'InputDose', 'Dose'}:
                row.update(Value=contract['dose_mg'], Unit='mg')
            elif field == 'Start time':
                row.update(Value=time, Unit='min')
            elif field == 'Infusion time':
                if contract['infusion_minutes'] is None:
                    raise Blocked('NONINFUSION_TARGET_HAS_SOURCE_INFUSION_OVERRIDE')
                row.update(Value=contract['infusion_minutes'], Unit='min')
            if template['canonical'] is None:
                row.pop('ValueOrigin', None)
            kept.append(row)
    sim['Parameters'] = kept
    reference['Name'] = new_name
    result['Protocols'].append(protocol)
    if result['Compounds'] != snapshot['Compounds'] or result.get('ExpressionProfiles') != snapshot.get('ExpressionProfiles'):
        raise ValueError('Regimen adapter changed drug or expression definitions')
    if result.get('Individuals') != snapshot.get('Individuals') or result.get('Formulations') != snapshot.get('Formulations'):
        raise ValueError('Regimen adapter changed physiology or formulation')
    actual_items, actual_times = _administrations(protocol)
    if actual_times != target or len(actual_items) != 1:
        raise ValueError('Registered native schedule differs from frozen clinical times')
    return {'snapshot': result, 'diff': _diff(snapshot, result), 'manifest': {
        'changed': True, 'source_protocol': old_name, 'new_protocol': new_name,
        'source_administration_times_min': old_times, 'explicit_times_min': target,
        'observation_origin_min': contract['observation_origin_min'],
        'drug_model_modified': False, 'clinical_outcomes_used': False,
        'vehicle_override_policy': 'Preserve uniform source vehicle/administration parameters at each explicit dose; clinical vehicle equivalence is not claimed.',
        'native_readback_still_required': True}}


def prepare_candidate(scenario, candidate, *, project_root=ROOT):
    """Produce a frozen executable plan or explicit operational blockers."""
    root = Path(project_root).resolve(strict=True)
    plan = {"version": VERSION, "status": "blocked", "candidate_id": candidate.get("candidate_id"),
        "execution_revision": EXECUTION_REVISION, "implementation_sha256": implementation_hashes(),
        "scenario_id": scenario.get("scenario_id", scenario.get("case_id")),
        "scenario_sha256": _digest(scenario), "candidate_sha256": _digest(candidate),
        "reasons": [], "scientific_inapplicability_inferred": False,
        "sealed_observations_accessed": False, "fitting_performed": False,
        "replacement_blueprint_constructed": False}
    try:
        contract = scenario_contract(scenario)
        plan["contract"] = contract
        if contract.get("salt_conversion"):
            conversion = contract["salt_conversion"]
            if _sha(_inside(conversion["source_file"], root)) != conversion["source_file_sha256"]:
                raise Blocked("DOSE_CONVERSION_SOURCE_CHECKSUM_MISMATCH")
        if not contract["case_id"] or not contract["compound"]:
            raise Blocked("CASE_AND_COMPOUND_ID_REQUIRED")
        registered = candidate.get("compounds", candidate.get("registered_compounds", []))
        if contract["compound"] not in {_name(c) for c in registered}:
            raise Blocked("TARGET_COMPOUND_NOT_REGISTERED_IN_CANDIDATE")
        prov = candidate.get("provenance", candidate.get("catalog_provenance", {}))
        if prov.get("license_status") != "allowed":
            raise Blocked("CANDIDATE_LICENSE_NOT_RESOLVED")
        original_path = _inside(prov.get("source_path", ""), root)
        if _sha(original_path) != prov.get("source_sha256"):
            raise Blocked("SOURCE_CHECKSUM_MISMATCH")
        path, execution_sha = original_path, prov["source_sha256"]
        if path.suffix == ".json" and path.stat().st_size <= 32 * 1024**2:
            snapshot = json.loads(path.read_text())
        else:
            snapshot = {}
        if path.suffix != ".json" or snapshot.get("Version") != 80:
            try:
                from recommendation_candidate_migration import resolve_migrated_source
                migration = resolve_migrated_source(prov, project_root=root)
            except (ValueError, OSError) as exc:
                raise Blocked("VERIFIED_NATIVE_MIGRATION_UNAVAILABLE:" + str(exc)) from exc
            path, execution_sha = migration["path"], migration["sha256"]
            plan["source_migration"] = {k:str(v) if isinstance(v,Path) else v for k,v in migration.items()}
            snapshot = json.loads(path.read_text())
        if path.stat().st_size > 32 * 1024**2:
            raise Blocked("SOURCE_EXCEEDS_32_MIB")
        if snapshot.get("Version") != 80:
            raise Blocked("NATIVE_SOURCE_NOT_VERSION_80")
        options, rejected = _source_options(snapshot, contract, source_sha256=prov["source_sha256"])
        plan["excluded_source_simulations"] = rejected
        if not options:
            raise Blocked("NO_SOURCE_SIMULATION_MATCHES_FIXED_EXECUTION_CONTRACT")
        chosen = options[0]
        source = load_source_model(path, execution_sha, chosen["name"], root)
        request = {"simulation_name": "candidate_" + _digest([contract["case_id"], candidate["candidate_id"]])[:16],
            "compound_name": chosen["compound_name"], "dose": {"value": contract["dose_mg"], "unit": "mg"},
            "output_selections": [chosen["output_path"]]}
        if contract["infusion_minutes"] is not None:
            request["infusion_minutes"] = contract["infusion_minutes"]
        # A prior native binding may select an existing source formulation only
        # when it names this exact immutable source; never invent dissolution.
        binding = contract["native_binding"]
        bound_source = binding.get("source_snapshot", {})
        if (contract["route"] == "Oral" and bound_source.get("sha256") == prov["source_sha256"]
                and binding.get("formulation_name")):
            request["formulation_name"] = binding["formulation_name"]
        adapted = adapt_source_model(source, request, root)
        if adapted["status"] != "adapted":
            raise Blocked("SOURCE_ADAPTER_BLOCKED:" + ";".join(adapted.get("reasons", [])))
        compound = _named(adapted["snapshot"].get("Compounds"), chosen["compound_name"], "Compounds")
        mw = _parameters(compound).get("Molecular weight", {})
        if mw.get("Unit") != "g/mol":
            raise Blocked("EXPLICIT_PARENT_MOLECULAR_WEIGHT_G_PER_MOL_REQUIRED")
        molecular_weight = _number(mw.get("Value"), "molecular weight", positive=True)
        prediction_contract = contract["prediction_contract"]
        # Source subjects remain exact source subjects. Clinical aggregate
        # populations are never reconstructed by guessing unknown covariates.
        old_permission = prediction_contract.get("source_reference_subject_prediction_allowed")
        approximation_allowed = old_permission is True
        if old_permission is not False:
            approximation_allowed |= scenario.get("execution_policy", {}).get("allow_reference_subject_approximation") is True
        if not approximation_allowed:
            raise Blocked("REFERENCE_SUBJECT_APPROXIMATION_REQUIRES_EXPLICIT_PROTOCOL_PERMISSION")
        approximate = ["source_reference_subject_is_not_clinical_group_statistic",
            "target_covariates_not_fitted_or_silently_substituted"]
        if contract["route"] == "Oral":
            approximate += ["source_formulation_not_verified_identical_to_clinical_product",
                "target_food_schedule_not_reconstructed"]
        regimen = apply_explicit_regimen(adapted['snapshot'], contract)
        if regimen['manifest']['changed']:
            approximate.append('source_administration_vehicle_parameters_reused_at_explicit_target_doses')
        # A longer explicit clinical schedule may require a larger original-grid
        # preservation budget. This bound depends on design alone, never outcomes.
        point_limit = 16384 if len(contract['administration_times_min']) > 1 else 4096
        gridded = ensure_output_times(regimen["snapshot"], contract["absolute_sampling_times_min"], root,
                                      max_points=point_limit)
        final, grid = gridded["snapshot"], gridded["manifest"]
        if final["Compounds"] != source["snapshot"]["Compounds"]:
            raise ValueError("Compound model was modified")
        # Target values come from the clinical design; only native paths come
        # from the selected model. These checks are independent of outcomes.
        expectations = derive_administration_expectations(final)
        target_times = contract['administration_times_min']
        if expectations["expected_event_count"] != len(target_times):
            raise Blocked("NATIVE_EVENT_COUNT_DIFFERS_FROM_EXPLICIT_CLINICAL_SCHEDULE")
        for row in expectations["expectations"]:
            field = row["path"].split("|")[-1]
            match = re.search(r'\|Application_([1-9][0-9]*)\|ProtocolSchemaItem\|', row['path'])
            if not match or int(match[1]) > len(target_times):
                raise Blocked('NATIVE_EVENT_PATH_NOT_BOUND_TO_TARGET_SCHEDULE')
            if field == "Dose":
                row.update(value=contract["dose_mg"], unit="mg")
            elif field == "Start time":
                row.update(value=target_times[int(match[1]) - 1], unit="min")
            elif field == "Infusion time":
                row.update(value=contract["infusion_minutes"], unit="min")
        plan.update(status="ready", source_path=str(original_path.relative_to(root)), source_sha256=prov["source_sha256"],
            execution_source_path=str(path.relative_to(root)), execution_source_sha256=execution_sha,
            source_simulation=chosen["name"], selection_rule="route+one-unambiguous-source-administration-item; fasted-name; log-dose-distance; source-name",
            request=request, output_path=chosen["output_path"], molecular_weight_g_per_mol=molecular_weight,
            compound_definition_sha256=_digest(source["snapshot"]["Compounds"]),
            source_individual=chosen["individual"], compatibility={"stratum": "reference_subject_approximation",
                "strict_target_match": False, "approximations": approximate,
                "drug_identity_match": True, "route_match": True, "dose_match_planned": True},
            snapshot=final, snapshot_sha256=_digest(final), grid_audit=grid,
            explicit_regimen_registration=regimen['manifest'], regimen_diff=regimen['diff'],
            native_expectations=expectations, source_adaptation_checks=adapted["checks"],
            source_adaptation_diff=adapted["diff"])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        plan["reasons"].append(type(exc).__name__ + ":" + str(exc))
    return plan


def _interpolate(times, values, target):
    import bisect
    if target < times[0] - 1e-8 or target > times[-1] + 1e-8:
        raise ValueError("Prediction would require extrapolation")
    index = bisect.bisect_left(times, target)
    if index < len(times) and abs(times[index] - target) <= 1e-8:
        return values[index]
    if index == 0 or index == len(times):
        raise ValueError("Requested time not covered by native curve")
    f = (target - times[index - 1]) / (times[index] - times[index - 1])
    return values[index - 1] * (1 - f) + values[index] * f


def prediction_from_curve(curve, plan):
    if curve.get("time_unit") != "min" or curve.get("unit") != "umol/l":
        raise ValueError("Exact native time and molar units are required")
    times, values = curve["time_min"], curve["concentration"]
    if len(times) != len(values) or len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Native time/concentration arrays differ or are not strictly increasing")
    if any(not math.isfinite(x) or x < 0 for x in times + values):
        raise ValueError("Native curve contains negative or nonfinite concentration/time")
    matrix = plan["contract"].get("matrix", "plasma")
    if matrix not in {"plasma", "whole_blood"}:
        raise ValueError("Unsupported native prediction matrix")
    if matrix == "whole_blood" and not re.fullmatch(
            r"Organism\|PeripheralVenousBlood\|[^|]+\|Whole Blood \(Peripheral Venous Blood\)", plan["output_path"]):
        raise ValueError("Whole blood prediction requires its exact source-native whole-blood observer")
    if matrix == "plasma" and 'Whole Blood (' in plan['output_path']:
        raise ValueError("Whole blood observer cannot be relabelled as a plasma prediction")
    mw = _number(plan["molecular_weight_g_per_mol"], "MW", positive=True)
    relative_required = plan["contract"]["sampling_times_min"]
    origin = plan['contract'].get('observation_origin_min', 0.)
    required = plan['contract'].get('absolute_sampling_times_min', [origin + t for t in relative_required])
    if required != [origin + t for t in relative_required]:
        raise ValueError('Absolute and relative sampling times disagree with frozen origin')
    sample = [_interpolate(times, values, t) * mw for t in required]
    end = required[-1]
    window = sorted(set([origin, end] + [t for t in times if origin < t < end]))
    window_values = [_interpolate(times, values, t) * mw for t in window]
    auc = sum((b - a) / 60 * (x + y) / 2 for a, b, x, y in
              zip(window, window[1:], window_values, window_values[1:]))
    sampling_registration = plan['contract'].get('sampling_grid_registration')
    cmax_key = 'dense_grid_Cmax' if sampling_registration else 'sampled_Cmax'
    return {"version": VERSION, "candidate_id": plan["candidate_id"], "scenario_id": plan["scenario_id"],
        "execution_revision": plan.get('execution_revision', EXECUTION_REVISION),
        "implementation_sha256": plan.get('implementation_sha256', implementation_hashes()),
        "native_output_path": plan["output_path"], "matrix": matrix, "analyte": "parent", "binding": "total",
        "time_h": [(t - origin) / 60 for t in times], "concentration_ng_ml": [v * mw for v in values],
        "time_origin_h_after_first_dose": origin / 60,
        "time_axis_definition": "hours relative to the frozen clinical observation origin; earlier native history retained at negative times",
        "native_time_h_after_first_dose": [t / 60 for t in times],
        "native_curve": curve, "molecular_weight_g_per_mol": mw,
        "mass_conversion": "umol/L times parent g/mol equals ug/L equals ng/mL",
        "sampling_time_h": [t / 60 for t in relative_required], "sampled_concentration_ng_ml": sample,
        "sampling_absolute_time_h_after_first_dose": [t / 60 for t in required],
        "sampling_grid_registration": deepcopy(sampling_registration),
        "endpoints": {cmax_key: {"value": max(sample), "unit": "ng/mL", "window_h": [relative_required[0]/60, (end-origin)/60]},
            "AUC_fixed_sampling_window": {"value": auc, "unit": "ng*h/mL", "window_h": [0., (end-origin)/60],
                "definition": "native registered-output-grid trapezoidal integral from the clinical observation origin; not individual last-quantifiable AUC"}},
        "uncomputed_endpoints": {"AUCinf": "No terminal extrapolation estimator registered in this adapter",
            "AUClast": "Individual last quantifiable time is not supplied by the outcome-free protocol"},
        "compatibility": plan["compatibility"], "prediction_accuracy_evaluated": False}


def execute_candidate(scenario, candidate, output_dir, *, project_root=ROOT, timeout_seconds=300):
    """Execute once and lock outputs, preserving every blocked/failed attempt."""
    root = Path(project_root).resolve(strict=True)
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be 1..600")
    output = _inside(output_dir, root, exists=False)
    if output.exists():
        raise FileExistsError("Never overwrite candidate prediction attempts")
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise OSError("At least 2 GiB free space required before candidate execution")
    output.mkdir(parents=True)
    plan = prepare_candidate(scenario, candidate, project_root=root)
    _save(output / "execution_plan.json", plan)
    result = {"version": VERSION, "scenario_id": plan["scenario_id"], "candidate_id": plan["candidate_id"],
        "execution_revision": plan['execution_revision'], "implementation_sha256": plan['implementation_sha256'],
        "status": plan["status"], "reasons": plan["reasons"], "native_executed": False,
        "prediction_locked": False, "sealed_observations_accessed": False,
        "fitting_performed": False, "scientific_inapplicability_inferred": False,
        "execution_plan_sha256": _sha(output / "execution_plan.json")}
    try:
        if plan["status"] == "ready":
            execution = execute_snapshot(snapshot=plan["snapshot"], output_dir=output / "native", project_root=root,
                timeout_seconds=timeout_seconds, max_output_bytes=128 * 1024**2, cpu_threads=2)
            result.update(native_executed=True, native_status=execution["status"],
                execution_manifest=execution["manifest_path"])
            if not execution.get("engineering_accepted"):
                result.update(status="native_failed", reasons=[str(execution.get("stop_reason", "native engineering verification failed"))])
            else:
                audit = audit_execution(snapshot=plan["snapshot"], execution_manifest=execution,
                    output_dir=output / "readback", project_root=root, **plan["native_expectations"])
                _save(output / "readback_result.json", audit)
                if audit.get("accepted") is not True:
                    result.update(status="readback_failed", reasons=audit.get("reasons", []))
                else:
                    curve = read_curve(output / "native", execution, plan["output_path"])
                    prediction = prediction_from_curve(curve, plan)
                    _save(output / "prediction.json", prediction)
                    files = [output / "execution_plan.json", output / "prediction.json", output / "readback_result.json",
                        _inside(execution["manifest_path"], root)]
                    for collection in ("pkml_outputs", "curve_outputs"):
                        files.extend(_inside(output / "native" / item["path"], root) for item in execution[collection])
                    lock = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                        "execution_revision": plan['execution_revision'], "implementation_sha256": plan['implementation_sha256'],
                        "scenario_id": plan["scenario_id"], "candidate_id": plan["candidate_id"],
                        "scenario_sha256": plan["scenario_sha256"], "candidate_sha256": plan["candidate_sha256"],
                        "source_sha256": plan["source_sha256"], "files": {str(p.relative_to(root)): _sha(p) for p in files},
                        "sealed_observations_accessed": False, "execution_verified": True,
                        "prediction_accuracy_evaluated": False, "compatibility": plan["compatibility"]}
                    if _sha(_inside(plan["source_path"], root)) != plan["source_sha256"]:
                        raise ValueError("Source changed during execution")
                    if implementation_hashes() != plan['implementation_sha256']:
                        raise ValueError('Execution implementation changed during prediction')
                    _save(output / "prediction_lock.json", lock)
                    result.update(status="prediction_locked", prediction_locked=True,
                        compatibility=plan["compatibility"], prediction_file=str((output / "prediction.json").relative_to(root)),
                        prediction_sha256=_sha(output / "prediction.json"),
                        prediction_lock=str((output / "prediction_lock.json").relative_to(root)),
                        prediction_lock_sha256=_sha(output / "prediction_lock.json"),
                        molecular_weight_g_per_mol=plan["molecular_weight_g_per_mol"],
                        source={"path": plan["source_path"], "sha256": plan["source_sha256"],
                            "simulation": plan["source_simulation"]},
                        strict_target_match=False, approximation_reason=plan["compatibility"]["approximations"],
                        curve=curve, predicted_endpoints=prediction["endpoints"])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result.update(status="failed", reasons=[type(exc).__name__ + ":" + str(exc)])
    _save(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--catalog", type=Path, default=Path("experiments/model_recommendation/frontier_core_145_20260921/inputs/candidates.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    scenario = json.loads(_inside(args.scenario).read_text())
    candidates = [json.loads(line) for line in _inside(args.catalog).read_text().splitlines() if line.strip()]
    matches = [c for c in candidates if c["candidate_id"] == args.candidate_id]
    if len(matches) != 1:
        raise ValueError("Exactly one frozen catalog candidate must match")
    result = execute_candidate(scenario, matches[0], args.output, timeout_seconds=args.timeout_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "prediction_locked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
