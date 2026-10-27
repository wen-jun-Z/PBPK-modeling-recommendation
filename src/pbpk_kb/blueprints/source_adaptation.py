"""Preserve a sourced PBPK simulation and apply explicit, bounded user changes.

This adapter does not fit parameters, inspect observation values, or execute a
solver. Native building blocks and simulation-local overrides remain intact.
Unsupported requests return a reason instead of silently reducing the model.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from .human_model import load_standard_human_model, STANDARD_HUMAN_MODEL_ID

VERSION = 'source-model-adaptation-v1.2'
_ROOT_NON_SOLVER = {
    'ObservedData', 'ObservedDataClassifications', 'ParameterIdentifications',
    'ParameterIdentificationClassifications', 'SimulationClassifications',
    'SimulationComparisons', 'SimulationComparisonClassifications',
    'QualificationPlans', 'QualificationPlanClassifications',
}
_SIM_NON_SOLVER = {'ObservedData', 'OutputMappings', 'IndividualAnalyses', 'PopulationAnalyses', 'Analyses'}
_REQUEST_FIELDS = {'simulation_name', 'compound_name', 'individual_name', 'protocol_name',
    'formulation_name', 'dose_scale', 'dose', 'infusion_minutes', 'output_selections'}
_GROUP_FIELDS = {'COMPOUND_LIPOPHILICITY': 'Lipophilicity',
    'COMPOUND_FRACTION_UNBOUND': 'FractionUnbound', 'COMPOUND_SOLUBILITY': 'Solubility',
    'COMPOUND_INTESTINAL_PERMEABILITY': 'IntestinalPermeability',
    'COMPOUND_PERMEABILITY': 'Permeability'}
_SYSTEMIC = {'GFR': 'GlomerularFiltration', 'Renal': 'KidneyClearance',
    'Hepatic': 'LiverClearance', 'Biliary': 'BiliaryClearance'}
_TIME = {'min': 1., 'h': 60., 's': 1./60., 'day(s)': 1440., 'd': 1440.}
_MASS = {'g': 1000., 'mg': 1., 'µg': .001, 'ug': .001, 'ng': .000001}


class SourceAdaptationError(ValueError):
    """An invalid or changed source must not enter an adaptation workflow."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _inside(path, root):
    path = Path(path)
    path = (path if path.is_absolute() else root / path).resolve(strict=True)
    if not path.is_relative_to(root) or path == root or not path.is_file():
        raise SourceAdaptationError('SOURCE_PATH_OUTSIDE_PROJECT_OR_NOT_FILE')
    return path


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SourceAdaptationError('DUPLICATE_JSON_KEY: ' + key)
        value[key] = item
    return value


def _named(rows, name, field):
    if not isinstance(rows, list):
        raise SourceAdaptationError('MISSING_BUILDING_BLOCK_LIST: ' + field)
    matches = [row for row in rows if isinstance(row, dict) and row.get('Name') == name]
    if len(matches) != 1:
        raise SourceAdaptationError(f'NONUNIQUE_OR_MISSING_REFERENCE: {field}[{name}]')
    return matches[0]


def _number(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value < 0 if zero else value <= 0)):
        raise SourceAdaptationError('INVALID_POSITIVE_NUMBER: ' + name)
    return float(value)


def _safe_name(value):
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= 160
        and not any(char in value for char in '/\\\x00\n\r') and value not in {'.', '..'})


def _parameters(container):
    rows = container.get('Parameters', [])
    if not isinstance(rows, list) or any(not isinstance(p, dict) or not isinstance(p.get('Name'), str) for p in rows):
        raise SourceAdaptationError('INVALID_NAMED_PARAMETERS')
    names = [p['Name'] for p in rows]
    if len(names) != len(set(names)):
        raise SourceAdaptationError('DUPLICATE_NAMED_PARAMETER')
    return {p['Name']: p for p in rows}


def _time(parameter):
    unit = parameter.get('Unit')
    if unit not in _TIME:
        raise SourceAdaptationError('UNSUPPORTED_TIME_UNIT: ' + str(unit))
    return _number(parameter.get('Value'), 'time', zero=True) * _TIME[unit]


def _mass_unit(unit):
    if not isinstance(unit, str):
        raise SourceAdaptationError('EXPLICIT_DOSE_UNIT_REQUIRED')
    weighted = unit.endswith('/kg')
    base = unit[:-3] if weighted else unit
    if base not in _MASS:
        raise SourceAdaptationError('UNSUPPORTED_DOSE_UNIT: ' + unit)
    return _MASS[base], weighted


def _dose_in(value, from_unit, to_unit):
    source, source_weighted = _mass_unit(from_unit)
    target, target_weighted = _mass_unit(to_unit)
    if source_weighted != target_weighted:
        raise SourceAdaptationError('DOSE_UNIT_REQUIRES_BODY_WEIGHT_CONVERSION')
    return _number(value, 'dose') * source / target


def _bind_requested_dose(parameter, requested):
    """Bind an explicit absolute prescription without inventing body weight.

    ApplicationFactory.SetApplicationDose chooses its mass branch from the
    InputDose unit. This changes the requested regimen, not drug physiology.
    Native event-dose readback must still verify the absolute target mass.
    """
    _, old_weighted = _mass_unit(parameter.get('Unit'))
    _, new_weighted = _mass_unit(requested['unit'])
    if old_weighted and not new_weighted:
        if parameter.get('TableFormula'):
            raise SourceAdaptationError('WEIGHT_DOSE_TABLE_CANNOT_BE_REPLACED')
        parameter['Unit'] = requested['unit']
        _update_value(parameter, _number(requested['value'], 'explicit absolute dose'))
    else:
        _update_value(parameter, _dose_in(requested['value'], requested['unit'], parameter.get('Unit')))


def _uniform_vehicle_items(items):
    """Require the same complete vehicle, route and formulation in every item.

    Source dose and time belong to the explicitly replaced regimen. All other
    parameters, including water, target organ and formulation key, must match.
    Within-schema split-input models are not accepted by _administrations.
    """
    if not items:
        raise SourceAdaptationError('SOURCE_ADMINISTRATION_ITEMS_REQUIRED')
    def vehicle(item):
        result = {k: deepcopy(v) for k, v in item.items()
                  if k not in {'Name', 'Description', 'Parameters'}}
        result['Parameters'] = sorted((deepcopy(p) for p in item.get('Parameters', [])
            if p.get('Name') not in {'InputDose', 'Start time'}), key=lambda p: p['Name'])
        return result
    if len(items) > 1 and any(p.get('TableFormula') for item in items for p in item.get('Parameters', []) if p.get('Name') in {'InputDose', 'Start time'}):
        raise SourceAdaptationError('DYNAMIC_SOURCE_DOSE_OR_TIME_CANNOT_BE_REPLICATED')
    first = vehicle(items[0])
    if any(vehicle(item) != first for item in items[1:]):
        raise SourceAdaptationError('NONUNIFORM_SOURCE_VEHICLE_CANNOT_BE_REPLICATED')
    return items[0]


def _profile_id(profile):
    return '|'.join(str(profile.get(key, '')) for key in ('Molecule', 'Species', 'Category'))


def _process_for(selection, compound):
    processes = compound.get('Processes', [])
    name, molecule = selection.get('Name'), selection.get('MoleculeName')
    if molecule:
        # PK-Sim selects Compound.ProcessByName(Name), then separately maps
        # the physiological MoleculeName. Author process names may therefore
        # use OATP1B1/1B3 while the selected human molecule is OATP1B1.
        # Resolve the authored name uniquely; never rewrite either molecule.
        matches = [p for p in processes if isinstance(p.get('Molecule'), str)
                   and p['Molecule'] and p['Molecule'] + '-' + str(p.get('DataSource')) == name]
    else:
        kind = _SYSTEMIC.get(selection.get('SystemicProcessType'))
        matches = [p for p in processes if p.get('InternalName') == kind
                   and isinstance(name, str) and name.endswith('-' + str(p.get('DataSource')))]
    if len(matches) != 1:
        raise SourceAdaptationError('UNBOUND_OR_AMBIGUOUS_PROCESS: ' + str(name))
    return matches[0]


def _one_administered_selection(simulation):
    """Identify one actually dosed parent without discarding selected metabolites.

    Reachability of the undosed selections is checked separately against the
    retained source metabolism definitions by ``_check_dependencies``.
    """
    selections = simulation.get('Compounds')
    if (not isinstance(selections, list) or not selections
            or any(not isinstance(row, dict) or not isinstance(row.get('Name'), str)
                   or not row['Name'] for row in selections)):
        raise SourceAdaptationError('INVALID_SELECTED_COMPOUND_BINDINGS')
    if len({row['Name'] for row in selections}) != len(selections):
        raise SourceAdaptationError('DUPLICATE_SELECTED_COMPOUND_BINDINGS')
    administered = [row for row in selections if row.get('Protocol')]
    if len(administered) != 1:
        raise SourceAdaptationError('MULTIDRUG_SIMULATION_REQUIRES_SEPARATE_DDI_ADAPTER')
    if not isinstance(administered[0]['Protocol'], dict):
        raise SourceAdaptationError('INVALID_SELECTED_PROTOCOL_BINDING')
    return administered[0]


def _check_dependencies(snapshot):
    simulations = snapshot.get('Simulations', [])
    if len(simulations) != 1:
        raise SourceAdaptationError('EXACTLY_ONE_SELECTED_SIMULATION_REQUIRED')
    simulation = simulations[0]
    if simulation.get('Model') != '4Comp' or simulation.get('Population'):
        raise SourceAdaptationError('ONLY_INDIVIDUAL_STANDARD_HUMAN_4COMP_SUPPORTED')
    individual = _named(snapshot.get('Individuals'), simulation.get('Individual'), 'Individuals')
    if individual.get('OriginData', {}).get('Species') != 'Human':
        raise SourceAdaptationError('SOURCE_INDIVIDUAL_MUST_BE_HUMAN')
    selection = _one_administered_selection(simulation)
    selections = simulation['Compounds']
    compound = _named(snapshot.get('Compounds'), selection.get('Name'), 'Compounds')
    selected_compounds = {row['Name']: _named(snapshot.get('Compounds'), row['Name'], 'Compounds')
                          for row in selections}
    if any(row.get('IsSmallMolecule') is not True for row in selected_compounds.values()):
        raise SourceAdaptationError('ONLY_SOURCE_SMALL_MOLECULE_SUPPORTED')
    profiles = {}
    for profile in snapshot.get('ExpressionProfiles', []):
        identifier = _profile_id(profile)
        if identifier in profiles:
            raise SourceAdaptationError('AMBIGUOUS_EXPRESSION_PROFILE: ' + identifier)
        profiles[identifier] = profile
    selected_profiles = individual.get('ExpressionProfiles', [])
    if len(selected_profiles) != len(set(selected_profiles)):
        raise SourceAdaptationError('DUPLICATE_INDIVIDUAL_EXPRESSION_PROFILE')
    for identifier in selected_profiles:
        if identifier not in profiles or profiles[identifier].get('Species') != 'Human':
            raise SourceAdaptationError('UNBOUND_HUMAN_EXPRESSION_PROFILE: ' + str(identifier))
    molecules = {profiles[identifier]['Molecule'] for identifier in selected_profiles}
    linked, disabled, metabolite_edges = [], [], []
    disabled_interactions, unselected_metabolite_defaults = [], []
    for selected in selections:
        selected_compound = selected_compounds[selected['Name']]
        for item in selected.get('Processes', []):
            # PK-Sim CompoundPropertiesMapper.notSelectedProcessFrom explicitly
            # represents disabled/partial choices with a molecule or systemic type
            # and no Name. Preserve them; do not invent or enable a process.
            if item.get('Name') in (None, ''):
                if item.get('MoleculeName'):
                    if item['MoleculeName'] not in molecules:
                        raise SourceAdaptationError('DISABLED_PROCESS_HAS_NO_HUMAN_EXPRESSION: ' + item['MoleculeName'])
                elif item.get('SystemicProcessType') not in _SYSTEMIC:
                    raise SourceAdaptationError('UNBOUND_DISABLED_PROCESS_SELECTION')
                disabled.append(deepcopy(item))
                continue
            process = _process_for(item, selected_compound)
            if item.get('MoleculeName') and item['MoleculeName'] not in molecules:
                raise SourceAdaptationError('ACTIVE_PROCESS_HAS_NO_HUMAN_EXPRESSION: ' + item['MoleculeName'])
            metabolite = process.get('Metabolite')
            selected_metabolite = item.get('MetaboliteName')
            # ProcessMappingMapper maps the simulation's MetaboliteName,
            # not the optional compound-process default. In particular, null
            # must not activate a default metabolite the author did not select.
            if selected_metabolite:
                if not isinstance(metabolite, str) or not metabolite:
                    raise SourceAdaptationError('METABOLITE_PROCESS_SELECTION_MISMATCH')
                _named(snapshot.get('Compounds'), selected_metabolite, 'Compounds')
                if selected_metabolite not in selected_compounds:
                    raise SourceAdaptationError('NAMED_METABOLITE_NEEDS_EXPLICIT_MULTICOMPOUND_BINDING')
                metabolite_edges.append({'parent': selected['Name'], 'metabolite': selected_metabolite,
                                         'process': item['Name']})
            elif metabolite:
                unselected_metabolite_defaults.append({'parent': selected['Name'],
                    'process': item['Name'], 'unselected_default': metabolite})
            linked.append(item['Name'])
        for alternative in selected.get('Alternatives', []):
            field = _GROUP_FIELDS.get(alternative.get('GroupName'))
            if field is None:
                raise SourceAdaptationError('UNREVIEWED_ALTERNATIVE_GROUP: ' + str(alternative.get('GroupName')))
            _named(selected_compound.get(field), alternative.get('AlternativeName'), field)
    reachable = {compound['Name']}
    while True:
        expanded = reachable | {edge['metabolite'] for edge in metabolite_edges if edge['parent'] in reachable}
        if expanded == reachable:
            break
        reachable = expanded
    if reachable != set(selected_compounds):
        raise SourceAdaptationError('UNDOSED_COMPOUND_NOT_AN_EXPLICIT_PARENT_METABOLITE')
    for item in simulation.get('Interactions', []):
        # SimulationMapper.findProcess uses an explicit NoInteractionProcess
        # for a molecule-only deselection. Preserve it without adding a drug,
        # resolving an active process, or enabling the interaction.
        if item.get('Name') in (None, ''):
            if (set(item) - {'Name', 'MoleculeName', 'CompoundName'} or
                    item.get('CompoundName') not in (None, '') or
                    item.get('MoleculeName') not in molecules):
                raise SourceAdaptationError('UNBOUND_DISABLED_INTERACTION_SELECTION')
            disabled_interactions.append(deepcopy(item))
            continue
        if item.get('CompoundName') not in selected_compounds:
            raise SourceAdaptationError('INTERACTION_REFERENCES_UNSELECTED_DRUG')
        _process_for(item, selected_compounds[item['CompoundName']])
        if item.get('MoleculeName') not in molecules:
            raise SourceAdaptationError('INTERACTION_HAS_NO_HUMAN_EXPRESSION: ' + str(item.get('MoleculeName')))
    protocol_ref = selection.get('Protocol', {})
    protocol = _named(snapshot.get('Protocols'), protocol_ref.get('Name'), 'Protocols')
    for formulation in protocol_ref.get('Formulations', []):
        _named(snapshot.get('Formulations'), formulation.get('Name'), 'Formulations')
    for field in ('Events', 'ObserverSets'):
        for reference in simulation.get(field, []):
            _named(snapshot.get(field), reference.get('Name'), field)
    for owner in (individual, simulation):
        paths = [p.get('Path') for p in owner.get('Parameters', [])]
        if any(not isinstance(p, str) or not p for p in paths) or len(paths) != len(set(paths)):
            raise SourceAdaptationError('INVALID_OR_DUPLICATE_LOCAL_OVERRIDE_PATH')
    return {'accepted': True, 'human_individual': individual['Name'], 'compound': compound['Name'],
        'protocol': protocol['Name'], 'active_processes': linked,
        'retained_undosed_metabolites': sorted(set(selected_compounds) - {compound['Name']}),
        'explicit_metabolite_links': metabolite_edges,
        'disabled_process_selections': disabled,
        'disabled_interaction_selections': disabled_interactions,
        'unselected_metabolite_defaults': unselected_metabolite_defaults,
        'interaction_count': len(simulation.get('Interactions', [])),
        'selected_expression_profiles': selected_profiles,
        'individual_override_count': len(individual.get('Parameters', [])),
        'simulation_override_count': len(simulation.get('Parameters', [])),
        'scope': 'source reference integrity; runtime parameter paths remain to be verified'}


def load_source_model(snapshot_path, sha256, simulation_name, project_root):
    """Verify source bytes and retain complete building blocks for one simulation.

    JSON parsing necessarily reads the source bytes, including any embedded
    observations. Observation values are never inspected for adaptation; those
    sections and their references are removed before returning the model.
    """
    root = Path(project_root).resolve(strict=True)
    path = _inside(snapshot_path, root)
    if not isinstance(sha256, str) or not re.fullmatch('[0-9a-f]{64}', sha256):
        raise SourceAdaptationError('EXPLICIT_SHA256_REQUIRED')
    if path.stat().st_size > 32 * 1024**2:
        raise SourceAdaptationError('SOURCE_EXCEEDS_32_MIB')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise SourceAdaptationError('SOURCE_CHECKSUM_MISMATCH')
    try:
        original = json.loads(raw, object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(SourceAdaptationError('NONFINITE_JSON: ' + value)))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SourceAdaptationError('INVALID_SOURCE_JSON') from exc
    if not isinstance(original, dict):
        raise SourceAdaptationError('SOURCE_SNAPSHOT_MUST_BE_OBJECT')
    selected = _named(original.get('Simulations'), simulation_name, 'Simulations')
    stripped = [{'path': '/' + key, 'reason': 'non_solver_observations_or_analysis'}
                for key in sorted(_ROOT_NON_SOLVER) if key in original]
    snapshot = deepcopy({key: value for key, value in original.items()
                         if key not in _ROOT_NON_SOLVER and key != 'Simulations'})
    simulation = deepcopy(selected)
    for key in sorted(_SIM_NON_SOLVER):
        if key in simulation:
            simulation.pop(key)
            stripped.append({'path': '/Simulations/0/' + key, 'reason': 'non_solver_observations_or_analysis'})
    simulation['HasResults'] = False
    snapshot['Simulations'] = [simulation]
    human = load_standard_human_model(root)
    source = {'source_path': str(path.relative_to(root)), 'source_sha256': sha256,
        'source_simulation_name': simulation_name, 'snapshot': snapshot,
        'snapshot_sha256': _digest(snapshot),
        'human_base': {'registry_id': STANDARD_HUMAN_MODEL_ID, 'checksum': human['checksum'],
            'source_definition_version': human['model']['source_version']},
        'stripped_non_solver_sections': stripped,
        'unselected_simulations_removed': len(original['Simulations']) - 1,
        'source_has_results_cleared': selected.get('HasResults', False),
        'building_block_policy': 'retain all source building blocks; verify selected dependencies',
        'observations_used_for_adaptation': False}
    return source


def _administrations(protocol):
    """Return mutable source dose items and the exact supported timing layout.

    Existing advanced source schemas are supported when they contain one schema
    with one repeated item. More complicated schemas remain reusable unchanged,
    but changing their administration requires a separate reviewed adapter.
    """
    if protocol.get('ApplicationType'):
        if protocol.get('Schemas') or protocol.get('DosingInterval') != 'Single':
            raise SourceAdaptationError('ONLY_SOURCE_SIMPLE_SINGLE_OR_ONE_ITEM_REPEATED_REGIMEN_CAN_CHANGE')
        parameters = _parameters(protocol)
        return [protocol], [_time(parameters['Start time'])]
    schemas = protocol.get('Schemas')
    if not isinstance(schemas, list) or not schemas or len(schemas) > 30:
        raise SourceAdaptationError('COMPLEX_SOURCE_REGIMEN_CAN_ONLY_BE_PRESERVED')
    if len(schemas) > 1:
        # Schema.ExpandedSchemaItems adds schema start and repeated interval;
        # ProtocolToSchemaItemsMapper sorts the expanded items by absolute time.
        # One item per schema avoids collapsing within-dose split-input models.
        all_items, expanded = [], []
        for schema in schemas:
            if not isinstance(schema, dict) or set(schema) - {'Name', 'Description', 'SchemaItems', 'Parameters'}:
                raise SourceAdaptationError('UNREVIEWED_SOURCE_SCHEMA_FIELDS')
            items = schema.get('SchemaItems')
            if not isinstance(items, list) or len(items) != 1:
                raise SourceAdaptationError('MULTI_ITEM_SOURCE_REGIMEN_CAN_ONLY_BE_PRESERVED')
            parameters = _parameters(schema)
            if set(parameters) - {'Start time', 'NumberOfRepetitions', 'TimeBetweenRepetitions'}:
                raise SourceAdaptationError('UNREVIEWED_SOURCE_SCHEMA_PARAMETERS')
            if any(p.get('TableFormula') for p in parameters.values()):
                raise SourceAdaptationError('DYNAMIC_SOURCE_SCHEMA_TIME_CANNOT_BE_REPLICATED')
            # These defaults are literal official SchemaFactory.Create values.
            start = _time(parameters.get('Start time', {'Value': 0., 'Unit': 'min'}))
            repetitions = _number(parameters.get('NumberOfRepetitions', {'Value': 1.})['Value'], 'repetitions')
            interval = _time(parameters.get('TimeBetweenRepetitions', {'Value': 0., 'Unit': 'min'}))
            if not repetitions.is_integer() or repetitions > 30 or (repetitions > 1 and interval <= 0):
                raise SourceAdaptationError('REPETITION_COUNT_OR_INTERVAL_UNSUPPORTED')
            item = items[0]
            item_start = _time(_parameters(item)['Start time'])
            all_items.append(item)
            expanded += [start + item_start + index * interval for index in range(int(repetitions))]
        if len(expanded) > 30:
            raise SourceAdaptationError('SOURCE_SCHEDULE_EXCEEDS_THIRTY_ADMINISTRATIONS')
        expanded.sort()
        if any(b <= a for a, b in zip(expanded, expanded[1:])):
            raise SourceAdaptationError('AMBIGUOUS_SIMULTANEOUS_SOURCE_ADMINISTRATIONS')
        _uniform_vehicle_items(all_items)
        return all_items, expanded
    schema = schemas[0]
    items = schema.get('SchemaItems')
    if not isinstance(items, list) or len(items) != 1:
        raise SourceAdaptationError('MULTI_ITEM_SOURCE_REGIMEN_CAN_ONLY_BE_PRESERVED')
    parameters = _parameters(schema)
    repetitions = _number(parameters['NumberOfRepetitions']['Value'], 'repetitions')
    if not repetitions.is_integer() or repetitions > 30:
        raise SourceAdaptationError('REPETITION_COUNT_MUST_BE_INTEGER_1_TO_30')
    interval = _time(parameters['TimeBetweenRepetitions'])
    if repetitions > 1 and interval <= 0:
        raise SourceAdaptationError('REPEATED_DOSES_REQUIRE_POSITIVE_INTERVAL')
    item_parameters = _parameters(items[0])
    start = _time(parameters['Start time']) + _time(item_parameters['Start time'])
    return items, [start + index * interval for index in range(int(repetitions))]


def _event_override(parameter, protocol_name):
    path = parameter.get('Path', '')
    parts = path.split('|')
    if parts[0] not in {'Events', 'Applications'}:
        return None
    if len(parts) < 5 or parts[1] != protocol_name:
        return None
    app = next((part for part in parts[2:] if re.fullmatch('Application_[1-9][0-9]*', part)), None)
    if app is None or parts[-2] != 'ProtocolSchemaItem':
        raise SourceAdaptationError('UNREVIEWED_ADMINISTRATION_OVERRIDE: ' + path)
    return {'parts': parts, 'application_index': int(app.split('_')[1]), 'parameter': parts[-1]}


def _update_value(parameter, value):
    _number(value, 'adapted administration value', zero=True)
    parameter['Value'] = value
    # A modified value must not continue claiming the author's old fitted origin.
    parameter.pop('ValueOrigin', None)


def _adapt_administration(snapshot, request):
    simulation = snapshot['Simulations'][0]
    selection = _one_administered_selection(simulation)
    reference = selection['Protocol']
    old_name = reference['Name']
    old_protocol = _named(snapshot['Protocols'], old_name, 'Protocols')
    target_name = request.get('protocol_name', old_name)
    protocol = _named(snapshot['Protocols'], target_name, 'Protocols')
    changed_protocol = target_name != old_name
    changing = bool({'protocol_name', 'dose_scale', 'dose', 'infusion_minutes', 'formulation_name'} & request.keys())
    if not changing:
        return
    items, times = _administrations(protocol)
    application_types = {item.get('ApplicationType') for item in items}
    if not application_types <= {'Intravenous', 'IntravenousBolus', 'Oral'} or len(application_types) != 1:
        raise SourceAdaptationError('UNSUPPORTED_OR_MIXED_ADMINISTRATION_TYPES')
    application_type = next(iter(application_types))
    if changed_protocol:
        old_items, old_times = _administrations(old_protocol)
        if len(items) > 1 or len(old_items) > 1:
            raise SourceAdaptationError('MULTISCHEMA_PROTOCOL_SWITCH_REQUIRES_EXPLICIT_REGIMEN')
        if ({i.get('ApplicationType') for i in old_items} != application_types or old_times != times):
            raise SourceAdaptationError('PROTOCOL_SWITCH_REQUIRES_SAME_ROUTE_AND_APPLICATION_TIMES')
        reference['Name'] = target_name
    old_formulations = [item['Name'] for item in reference.get('Formulations', [])]
    if 'formulation_name' in request:
        if application_type != 'Oral' or len(reference.get('Formulations', [])) != 1:
            raise SourceAdaptationError('FORMULATION_CHANGE_REQUIRES_SINGLE_SOURCE_ORAL_BINDING')
        _named(snapshot.get('Formulations'), request['formulation_name'], 'Formulations')
        reference['Formulations'][0]['Name'] = request['formulation_name']
    if 'infusion_minutes' in request and application_type != 'Intravenous':
        raise SourceAdaptationError('INFUSION_CHANGE_REQUIRES_SOURCE_INTRAVENOUS_INFUSION')
    for item in items:
        parameters = _parameters(item)
        if 'dose_scale' in request:
            parameter = parameters['InputDose']
            _mass_unit(parameter.get('Unit'))
            _update_value(parameter, _number(parameter.get('Value'), 'source dose') * request['dose_scale'])
        if 'dose' in request:
            parameter = parameters['InputDose']
            _bind_requested_dose(parameter, request['dose'])
        if 'infusion_minutes' in request:
            parameter = parameters.get('Infusion time')
            if parameter is None or parameter.get('Unit') not in _TIME:
                raise SourceAdaptationError('EXPLICIT_SOURCE_INFUSION_TIME_REQUIRED')
            _update_value(parameter, request['infusion_minutes'] / _TIME[parameter['Unit']])
    # Source-local dose, infusion and water overrides have higher precedence
    # than the protocol. Explicitly synchronize them, including renamed paths.
    parameter_values = _parameters(items[0])
    for parameter in simulation.get('Parameters', []):
        override = _event_override(parameter, old_name)
        if override is None:
            if (parameter.get('Path', '').split('|')[-1] == 'InputDose'
                    and ({'dose', 'dose_scale', 'protocol_name'} & request.keys())):
                raise SourceAdaptationError('UNBOUND_LOCAL_DOSE_OVERRIDE: ' + parameter['Path'])
            continue
        if override['application_index'] > len(times):
            raise SourceAdaptationError('OVERRIDE_REFERENCES_NONEXISTENT_APPLICATION')
        key = override['parameter']
        if changed_protocol:
            if key not in parameter_values:
                raise SourceAdaptationError('PROTOCOL_SWITCH_CANNOT_MAP_LOCAL_OVERRIDE: ' + key)
            source_parameter = parameter_values[key]
            if key == 'InputDose':
                value = _dose_in(source_parameter['Value'], source_parameter.get('Unit'), parameter.get('Unit'))
            elif key == 'Start time':
                if parameter.get('Unit') not in _TIME:
                    raise SourceAdaptationError('LOCAL_START_TIME_UNIT_REQUIRED')
                value = times[override['application_index'] - 1] / _TIME[parameter['Unit']]
            elif source_parameter.get('Unit') in _TIME and parameter.get('Unit') in _TIME:
                value = _time(source_parameter) / _TIME[parameter['Unit']]
            elif source_parameter.get('Unit') == parameter.get('Unit'):
                value = source_parameter['Value']
            else:
                raise SourceAdaptationError('PROTOCOL_OVERRIDE_UNIT_CONFLICT: ' + key)
            _update_value(parameter, value)
            override['parts'][1] = target_name
        elif key == 'InputDose' and 'dose_scale' in request:
            _mass_unit(parameter.get('Unit'))
            _update_value(parameter, _number(parameter.get('Value'), 'local dose') * request['dose_scale'])
        elif key == 'InputDose' and 'dose' in request:
            _bind_requested_dose(parameter, request['dose'])
        elif key == 'Infusion time' and 'infusion_minutes' in request:
            if parameter.get('Unit') not in _TIME:
                raise SourceAdaptationError('LOCAL_INFUSION_UNIT_REQUIRED')
            _update_value(parameter, request['infusion_minutes'] / _TIME[parameter['Unit']])
        if 'formulation_name' in request:
            override['parts'] = [request['formulation_name'] if part in old_formulations else part
                                 for part in override['parts']]
        parameter['Path'] = '|'.join(override['parts'])


def _diff(before, after, path=''):
    if isinstance(before, dict) and isinstance(after, dict):
        rows = []
        for key in sorted(before.keys() | after.keys()):
            location = path + '/' + key.replace('~', '~0').replace('/', '~1')
            if key not in before or key not in after:
                rows.append({'path': location, 'before': before.get(key), 'after': after.get(key)})
            else:
                rows.extend(_diff(before[key], after[key], location))
        return rows
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [row for index, pair in enumerate(zip(before, after, strict=True))
                for row in _diff(*pair, path + '/' + str(index))]
    return [] if before == after else [{'path': path or '/', 'before': before, 'after': after}]


def adapt_source_model(source, request, project_root):
    """Apply explicit same-drug source-supported changes without numerical fitting.

    Unsupported requests return ``blocked`` with no executable snapshot. Source
    tampering and unsafe filesystem inputs raise ``SourceAdaptationError``.
    Existing repeated source protocols are retained as full native schemas;
    changing timing/topology, adding drugs or inventing physiology is excluded.
    """
    root = Path(project_root).resolve(strict=True)
    verified = load_source_model(source['source_path'], source['source_sha256'],
                                 source['source_simulation_name'], root)
    if _digest(source['snapshot']) != source['snapshot_sha256']:
        raise SourceAdaptationError('IMPORTED_SNAPSHOT_CHANGED')
    if source['snapshot_sha256'] != verified['snapshot_sha256']:
        raise SourceAdaptationError('IMPORTED_SNAPSHOT_DIFFERS_FROM_VERIFIED_SOURCE')
    if source['human_base'] != verified['human_base']:
        raise SourceAdaptationError('REGISTERED_HUMAN_BASE_CHANGED')
    source = verified
    manifest = {key: deepcopy(value) for key, value in verified.items() if key != 'snapshot'}
    manifest.update(adapter_version=VERSION, fitting_performed=False, model_executed=False,
        scientific_validated=False, request=deepcopy(request),
        limitations=['One administered source small molecule with its explicitly bound undosed metabolites and individual simulation only.',
            'Native field/observer paths and physiological suitability require execution and use-specific validation.',
            'Source fitted parameters are retained; this is same-drug adaptation, not de novo model discovery.',
            'New drugs, new DDI mechanisms, new organs, arbitrary physiology and new regimen timing are not generated.'])
    result = {'status': 'blocked', 'manifest': manifest, 'diff': [], 'checks': {}, 'reasons': []}
    try:
        if not isinstance(request, dict):
            raise SourceAdaptationError('REQUEST_MUST_BE_OBJECT')
        unknown = sorted(request.keys() - _REQUEST_FIELDS)
        if unknown:
            raise SourceAdaptationError('UNSUPPORTED_REQUEST_FIELDS: ' + ', '.join(unknown))
        if 'dose' in request and 'dose_scale' in request:
            raise SourceAdaptationError('DOSE_AND_DOSE_SCALE_ARE_MUTUALLY_EXCLUSIVE')
        for key in ('dose_scale', 'infusion_minutes'):
            if key in request:
                _number(request[key], key)
        if 'dose' in request:
            dose = request['dose']
            if not isinstance(dose, dict) or set(dose) != {'value', 'unit'}:
                raise SourceAdaptationError('DOSE_REQUIRES_EXACT_VALUE_AND_UNIT')
            _number(dose['value'], 'dose')
            _mass_unit(dose['unit'])
        for key in ('simulation_name', 'compound_name', 'individual_name', 'protocol_name', 'formulation_name'):
            if key in request and (not isinstance(request[key], str) or not request[key].strip()):
                raise SourceAdaptationError('NONEMPTY_NAME_REQUIRED: ' + key)
        snapshot = deepcopy(source['snapshot'])
        if type(snapshot.get('Version')) is not int or snapshot['Version'] != 80:
            raise SourceAdaptationError('SNAPSHOT_VERSION_REQUIRES_REAL_MIGRATION_TO_80')
        before_checks = _check_dependencies(snapshot)
        simulation = snapshot['Simulations'][0]
        if request.get('compound_name', before_checks['compound']) != before_checks['compound']:
            raise SourceAdaptationError('NEW_DRUG_REQUIRES_ITS_OWN_SOURCE_MODEL')
        if 'simulation_name' in request:
            simulation['Name'] = request['simulation_name']
        if not _safe_name(simulation.get('Name')):
            raise SourceAdaptationError('UNSAFE_OUTPUT_SIMULATION_NAME')
        if 'individual_name' in request and request['individual_name'] != simulation['Individual']:
            _named(snapshot.get('Individuals'), request['individual_name'], 'Individuals')
            for override in simulation.get('Parameters', []):
                parts = override['Path'].split('|')
                if parts[0] not in {'Events', 'Applications'} and before_checks['compound'] not in parts:
                    raise SourceAdaptationError('INDIVIDUAL_CHANGE_CONFLICTS_WITH_LOCAL_PHYSIOLOGY_OVERRIDE: ' + override['Path'])
            simulation['Individual'] = request['individual_name']
        if 'output_selections' in request:
            outputs = request['output_selections']
            if (not isinstance(outputs, list) or not outputs or any(not isinstance(x, str) for x in outputs)
                    or len(outputs) != len(set(outputs)) or not set(outputs) <= set(simulation.get('OutputSelections', []))):
                raise SourceAdaptationError('OUTPUTS_MUST_BE_NONEMPTY_SUBSET_OF_SOURCE_SIMULATION_OUTPUTS')
            simulation['OutputSelections'] = list(outputs)
        _adapt_administration(snapshot, request)
        checks = _check_dependencies(snapshot)
        if len(json.dumps(snapshot, ensure_ascii=False, allow_nan=False).encode()) > 8 * 1024**2:
            raise SourceAdaptationError('ADAPTED_SNAPSHOT_EXCEEDS_NATIVE_8_MIB_LIMIT')
        difference = _diff(source['snapshot'], snapshot)
        checks.update(source_checksum_verified=True, registered_human_base_verified=True,
            compound_definition_unchanged=snapshot['Compounds'] == source['snapshot']['Compounds'],
            expression_profiles_unchanged=snapshot.get('ExpressionProfiles') == source['snapshot'].get('ExpressionProfiles'),
            process_selections_unchanged=[row.get('Processes') for row in simulation['Compounds']] == [row.get('Processes') for row in source['snapshot']['Simulations'][0]['Compounds']],
            interactions_unchanged=simulation.get('Interactions') == source['snapshot']['Simulations'][0].get('Interactions'),
            source_fitted_parameters_retained=True, observation_sections_excluded=True,
            runtime_execution_verified=False, requested_physiology_scientifically_validated=False)
        manifest['adapted_snapshot_sha256'] = _digest(snapshot)
        manifest['request_change_count'] = len(difference)
        result.update(status='adapted', snapshot=snapshot, diff=difference, checks=checks)
    except (SourceAdaptationError, KeyError, TypeError) as exc:
        result['reasons'] = [str(exc)]
    return result
