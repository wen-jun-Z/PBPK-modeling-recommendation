"""Source-bound MM/low-concentration first-order alternatives on one human base.

Only source MM branches can be linearized. Returning to MM means choosing the
original sourced candidate; the adapter never invents Km for a first-order base.
No concentration observations or final labels are consumed here.
"""
from __future__ import annotations
from copy import deepcopy
import math
from pathlib import Path
import sqlite3
from uuid import NAMESPACE_URL, uuid5

from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.human_model import load_standard_human_model
from pbpk_kb.blueprints.human_model_snapshot import _INTERACTION_PROCESS_PARAMETERS
from pbpk_kb.blueprints.native_edits import (
    PARTITION_ACTIONS, _checksum, _path_differences, _snapshot_from_design,
    compile_candidate, partition_candidates, reviewed_process_parameter_path,
)
from pbpk_kb.blueprints.schemas import ProvenanceRef

ACTION = 'mm_low_concentration_first_order'
PROCESS_PAIRS = {
    'MetabolizationLiverMicrosomes_MM': 'MetabolizationLiverMicrosomes_FirstOrder',
    'rCYP450_MM': 'rCYP450_FirstOrder',
    'MetabolizationSpecific_MM': 'MetabolizationSpecific_FirstOrder',
}
CONCENTRATION_FACTORS = {'pmol/l': 1e-6, 'nmol/l': 1e-3, 'µmol/l': 1.,
                         'µM': 1., 'mM': 1000., 'mmol/l': 1000.}
NATIVE_UNIT = 'l/µmol/min'
MECHANISM_ADAPTER_VERSION = 'source-mm-low-concentration-1.0'


class MechanismNotApplicable(ValueError):
    """The source does not admit this particular reviewed mechanism edit."""


def _positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def km_umol_per_l(value, unit):
    if not _positive(value) or unit not in CONCENTRATION_FACTORS:
        raise ValueError('Km requires a finite positive source concentration and reviewed molar unit')
    converted = value * CONCENTRATION_FACTORS[unit]
    if not _positive(converted):
        raise ValueError('Km conversion overflow or underflow')
    return converted


def _identity(process, parameter):
    return {'internal_name': process['InternalName'], 'molecule': process.get('Molecule'),
            'data_source': process['DataSource'], 'parameter_name': parameter}


def _mechanism_source(project_root):
    """Bind both equations AND amount/concentration aliases to the locked DB."""
    definition = load_standard_human_model(project_root)
    database = [r for r in definition['resources'] if r['role'] == 'database']
    if len(database) != 1:
        raise ValueError('One locked equation database is required')
    resource = database[0]
    path = (project_root / resource['path']).resolve(strict=True)
    if not path.is_relative_to(project_root):
        raise ValueError('Mechanism database leaves project')
    formulas = {
        'MetabolizationSpecific_MM': ('EM * kcat * K_water * C / (Km + K_water * C)', 'Amount per time'),
        'Metabolization_FirstOrder': ('CP * CLspecPerEnzyme * M * K_water', 'Amount per time'),
        'PARAM_Apparent_kcat_Reaction': ('kcat * kcat_interaction_factor', 'Inversed time'),
        'PARAM_Apparent_Km_Reaction': ('Km * Km_interaction_factor', 'Concentration (molar)'),
        'PARAM_Apparent_CLSpecPerEnzyme': ('CLspecPerEnzyme * CLspecPerEnzyme_interaction_factor', 'Second order rate constant'),
    }
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
        for rate, expected in formulas.items():
            rows = connection.execute('SELECT formula,dimension FROM tab_calculation_method_rates WHERE calculation_method=? AND formula_rate=?', ('LinksCommon', rate)).fetchall()
            if rows != [expected]:
                raise ValueError('Native equation source drift: ' + rate)
        for mm, linear in PROCESS_PAIRS.items():
            for process, rate in ((mm, 'MetabolizationSpecific_MM'), (linear, 'Metabolization_FirstOrder')):
                if connection.execute('SELECT calculation_method,formula_rate FROM tab_process_rates WHERE process=?', (process,)).fetchall() != [('LinksCommon', rate)]:
                    raise ValueError('Native kinetic process mapping drift')
        aliases = {
            'MetabolizationSpecific_MM': {(370, 'EM', 1), (372, 'C', 0)},
            'Metabolization_FirstOrder': {(370, 'CP', 0), (372, 'M', 1)},
        }
        for rate, expected in aliases.items():
            rows = connection.execute('SELECT path_id,alias,use_amount FROM tab_rate_generic_molecules WHERE calculation_method=? AND formula_rate=?', ('LinksCommon', rate)).fetchall()
            if set(rows) != expected:
                raise ValueError('Native enzyme/substrate amount-concentration mapping drift')
    location = 'LinksCommon: MetabolizationSpecific_MM, Metabolization_FirstOrder; app parameters; molecule paths 370/372'
    source = {'provenance_id': str(uuid5(NAMESPACE_URL, resource['path'] + '#' + location)),
              'provenance_kind': 'model_asset', 'source_id': resource['path'],
              'source_location': location, 'source_checksum': resource['sha256']}
    return source, definition


def _branches(compound):
    branches = []
    seen = set()
    for index, process in enumerate(compound['Processes']):
        internal = process.get('InternalName')
        if internal not in PROCESS_PAIRS:
            continue
        if process.get('Species') != 'Human':
            raise MechanismNotApplicable('MM linearization requires sourced Human processes')
        parameters = {p['Name']: p for p in process['Parameters']}
        if len(parameters) != len(process['Parameters']) or not {'kcat', 'Km'} <= parameters.keys():
            raise MechanismNotApplicable('Each transformed MM branch requires explicit unique source kcat and Km')
        kcat, km = parameters['kcat'], parameters['Km']
        if kcat.get('Unit') != '1/min' or not _positive(kcat.get('Value')):
            raise MechanismNotApplicable('MM source kcat must be finite positive 1/min')
        normalized = km_umol_per_l(km['Value'], km.get('Unit'))
        if not _positive(kcat['Value'] / normalized) or not _positive(1. / normalized):
            raise ValueError('MM-to-linear derived clearance or conversion multiplier overflow')
        old_identity = _identity(process, 'kcat')
        reviewed_process_parameter_path(compound['Name'], old_identity)
        new_identity = {**old_identity, 'internal_name': PROCESS_PAIRS[internal], 'parameter_name': 'CLspec/[Enzyme]'}
        key = (process['Molecule'], process['DataSource'])
        if key in seen:
            raise MechanismNotApplicable('Native branch runtime identity is not unique')
        seen.add(key)
        branches.append({'process_index': index, 'from_identity': old_identity, 'to_identity': new_identity,
                         'kcat_anchor': kcat['Value'], 'km_source': deepcopy(km),
                         'km_umol_per_l': normalized, 'linear_anchor': kcat['Value'] / normalized,
                         'original_process': deepcopy(process)})
    if not branches:
        raise MechanismNotApplicable('No source MM branch with a reviewed first-order counterpart')
    enzymes = {b['from_identity']['molecule'] for b in branches}
    if any(p.get('InternalName') in _INTERACTION_PROCESS_PARAMETERS and p.get('Molecule') in enzymes
           for p in compound['Processes']):
        raise MechanismNotApplicable('Explicit same-enzyme interaction needs a separately audited approximation')
    return branches


def _convert_specs(compound, specs, branches, source):
    result = deepcopy(specs)
    if result is None:
        return None
    by_identity = {tuple(b['from_identity'][k] for k in ('internal_name','molecule','data_source','parameter_name')): b for b in branches}
    def resolve(identity):
        key = tuple(identity[k] for k in ('internal_name','molecule','data_source','parameter_name'))
        if key not in by_identity:
            raise ValueError('Fitted MM parameter is outside the transformed source branches')
        return by_identity[key]
    converted_paths = {reviewed_process_parameter_path(compound['Name'], b['from_identity']) for b in branches}
    for spec in result:
        pid = spec['parameter_id']
        if pid not in {'shared_process_parameter', 'single_process_kcat'}:
            if any(m.get('runtime_path') in converted_paths for m in spec.get('runtime_mappings', [])):
                raise ValueError('Unknown fitted kcat contract cannot cross the mechanism edit')
            continue
        ProvenanceRef.model_validate(spec.get('source'))
        mappings = spec['runtime_mappings']
        selected = [resolve(x) for x in spec['native_processes']] if pid == 'shared_process_parameter' else [resolve(spec['native_process'])]
        if len(mappings) != len(selected):
            raise ValueError('Mechanism conversion requires exact source mappings')
        for i, (mapping, branch) in enumerate(zip(mappings, selected, strict=True)):
            if (mapping.get('runtime_path') != reviewed_process_parameter_path(compound['Name'], branch['from_identity'])
                    or mapping.get('native_unit', spec['unit']) != '1/min'):
                raise ValueError('Mechanism conversion source kcat mapping drift')
            if pid == 'shared_process_parameter':
                if (spec.get('parameterization') != 'shared_factor' or spec['unit'] != 'dimensionless'
                        or spec['source_anchor_values'][i] != branch['kcat_anchor']
                        or mapping.get('multiplier') != branch['kcat_anchor']):
                    raise ValueError('Shared factor must retain original branch-specific kcat anchors')
                multiplier = branch['linear_anchor']
            else:
                if spec['unit'] != '1/min' or mapping.get('multiplier') != 1.:
                    raise ValueError('Single kcat must retain a physical 1/min fitting coordinate')
                multiplier = 1. / branch['km_umol_per_l']
            mapping.update(runtime_path=reviewed_process_parameter_path(compound['Name'], branch['to_identity']),
                           multiplier=multiplier, native_unit=NATIVE_UNIT)
        if pid == 'shared_process_parameter':
            if len(selected) < 2 or len({b['from_identity']['molecule'] for b in selected}) != 1:
                raise ValueError('Shared factor must not conflate distinct enzymes')
            spec['native_processes'] = [deepcopy(b['to_identity']) for b in selected]
            spec['source_anchor_values'] = [b['linear_anchor'] for b in selected]
        else:
            spec['native_process'] = deepcopy(selected[0]['to_identity'])
        spec['mechanism_conversion'] = {'action': ACTION, 'source': source,
                                        'branches': [deepcopy(b) for b in selected],
                                        'fitting_coordinate_preserved': True}
    return result


def _compile_native_candidate(request, action, *, project_root, parameter_specs=None):
    """Compose a partition action and the single reviewed kinetic approximation."""
    if action in PARTITION_ACTIONS:
        return compile_candidate(request, action, project_root=project_root, parameter_specs=parameter_specs)
    suffix = '__' + ACTION
    if not isinstance(action, str) or not action.endswith(suffix) or action[:-len(suffix)] not in PARTITION_ACTIONS:
        raise ValueError('Unsupported composed native mechanism action')
    root = Path(project_root).resolve(strict=True)
    parent = compile_candidate(request, action[:-len(suffix)], project_root=root, parameter_specs=parameter_specs)
    revised = deepcopy(parent['request'])
    compound = revised['instance']['compound']
    branches = _branches(compound)
    source, definition = _mechanism_source(root)
    if revised['human_model']['definition_checksum'] != definition['checksum']:
        raise ValueError('Mechanism definition checksum differs from source lock')
    converted_specs = _convert_specs(compound, parameter_specs, branches, source)
    compound_source = deepcopy(revised['instance']['compound_sources']['Processes'])
    for branch in branches:
        process = compound['Processes'][branch['process_index']]
        process['InternalName'] = branch['to_identity']['internal_name']
        retained = [p for p in process['Parameters'] if p['Name'] in {'Content of CYP proteins in liver microsomes', 'Enzyme concentration'}]
        process['Parameters'] = retained + [{'Name': 'CLspec/[Enzyme]', 'Value': branch['linear_anchor'], 'Unit': NATIVE_UNIT}]
    revised['instance']['compound_sources']['Processes'] = source
    design, snapshot = _snapshot_from_design(BlueprintDesignRequest.model_validate(revised), root)
    differences = _path_differences(parent['snapshot'], snapshot)
    prefixes = tuple(prefix + str(b['process_index']) + '/' for b in branches
                     for prefix in ('/Compounds/0/Processes/', '/Simulations/0/Compounds/0/Processes/'))
    if not differences or any(not d['path'].startswith(prefixes) for d in differences):
        raise ValueError('Mechanism edit unexpectedly changed human physiology, administration or other processes')
    condition = {'kind': 'low_local_unbound_substrate', 'dimensionless_ratio': 'K_water*C/Km',
                 'local_rate_relative_excess_of_linear_over_MM': 'K_water*C/Km',
                 'recommended_maximum_ratio_for_10_percent_local_rate_error': .1,
                 'qualified_at_all_doses': False, 'requires_scenario_specific_check': True,
                 'same_enzyme_interactions_supported': False,
                 'systemic_plasma_concentration_is_not_a_substitute_for_local_unbound_concentration': True}
    manifest = {'adapter_version': MECHANISM_ADAPTER_VERSION, 'action': action,
                'change_class': 'enzyme_kinetic_mechanism', 'topology_changed': False,
                'mechanism_changed': True, 'human_core_preserved': True, 'administration_preserved': True,
                'source': source, 'source_binding_verified': True, 'parent_process_source': compound_source,
                'branches': branches, 'applicability_condition': condition,
                'partition_edit': deepcopy(parent['edit_manifest']), 'pathdiff': differences,
                'snapshot_sha256': _checksum(snapshot), 'parent_snapshot_sha256': _checksum(parent['snapshot']),
                'compile_status': 'not_run', 'simulation_status': 'not_run', 'scientific_status': 'not_evaluated',
                'execution': {'solver_calls': 0, 'gpu_jobs': 0, 'llm_calls': 0},
                'limitations': ['The native rate law changes; anatomy and branch enzyme identities do not.',
                                'Linearization is a candidate hypothesis, not asserted valid at every dose.',
                                'Source kcat and Km may be historically fitted; independent validation remains necessary.']}
    return {'structure_id': action, 'request': revised, 'design': design, 'snapshot': snapshot,
            'parameter_specs': converted_specs, 'edit_count': parent['edit_count'] + 1,
            'semantic_hash': _checksum({'partition': parent['semantic_hash'], 'mechanism': ACTION,
                                       'branches': [b['to_identity'] for b in branches]}),
            'parameterization_hash': _checksum({'parent': parent['parameterization_hash'], 'specs': converted_specs,
                                               'branch_conversion': branches}), 'edit_manifest': manifest}


def compile_native_candidate(request, action, *, project_root, parameter_specs=None,
                             fitted_values=None, parameter_provenance=None):
    """Compile once from original anchors, optionally materialize fitted values.

    The source contract supplies the fitting coordinates. Its derived native
    contract supplies the path and unit for this candidate's actual rate law.
    Applying fitted values never runs the mechanism conversion a second time.
    """
    candidate = _compile_native_candidate(request, action, project_root=project_root,
                                          parameter_specs=parameter_specs)
    if fitted_values is None:
        return candidate
    if parameter_specs is None or parameter_provenance is None:
        raise ValueError('Fitted native rebuild requires original specs and explicit calibration provenance')
    from pbpk_kb.blueprints.native_edits import apply_parameter_values
    revised = apply_parameter_values(candidate['request'], candidate['parameter_specs'],
                                     fitted_values, parameter_provenance)
    design, snapshot = _snapshot_from_design(BlueprintDesignRequest.model_validate(revised), Path(project_root).resolve())
    candidate['edit_manifest']['parameter_application_pathdiff'] = _path_differences(candidate['snapshot'], snapshot)
    candidate['edit_manifest']['parameter_application_source'] = deepcopy(parameter_provenance)
    candidate['edit_manifest']['snapshot_sha256'] = _checksum(snapshot)
    candidate['edit_manifest']['fitted_values_materialized'] = True
    candidate['parameterization_hash'] = _checksum({'original': candidate['parameterization_hash'], 'fitted_values': fitted_values})
    candidate.update(request=revised, design=design, snapshot=snapshot)
    return candidate


def native_candidates(request, *, project_root, parameter_specs=None):
    candidates = partition_candidates(request, project_root=project_root, parameter_specs=parameter_specs)
    payload = request.model_dump(mode='json') if isinstance(request, BlueprintDesignRequest) else request
    try:
        _branches(payload['instance']['compound'])
    except MechanismNotApplicable:
        return candidates
    return candidates + [compile_native_candidate(request, action + '__' + ACTION,
                          project_root=project_root, parameter_specs=parameter_specs) for action in PARTITION_ACTIONS]


def apply_linearized_parameter(compound, spec, value):
    """Apply only adapter-derived kcat-coordinate contracts to first-order paths."""
    ProvenanceRef.model_validate(spec.get('source'))
    conversion = spec['mechanism_conversion']
    ProvenanceRef.model_validate(conversion.get('source'))
    branches = conversion.get('branches')
    shared = spec['parameter_id'] == 'shared_process_parameter'
    identities = spec.get('native_processes') if shared else [spec.get('native_process')]
    mappings = spec.get('runtime_mappings')
    if (conversion.get('action') != ACTION or not isinstance(branches, list) or not branches
            or not isinstance(identities, list) or len(branches) != len(identities)
            or len(branches) != len(mappings) or value < 0
            or spec['unit'] != ('dimensionless' if shared else '1/min')):
        raise ValueError('Invalid typed MM-to-linear parameter conversion')
    if spec['lower_bound'] < 0:
        raise ValueError('MM-to-linear source capacity bounds must be nonnegative')
    if shared and (spec.get('parameterization') != 'shared_factor' or not 2 <= len(branches) <= 8
                   or len({b['from_identity']['molecule'] for b in branches}) != 1):
        raise ValueError('Linear shared factor needs independent anchors of the same enzyme')
    if not shared and len(branches) != 1:
        raise ValueError('A single physical kcat coordinate cannot bind several enzymes')
    updates, seen = [], set()
    for index, (branch, identity, mapping) in enumerate(zip(branches, identities, mappings, strict=True)):
        old = branch['from_identity']
        expected = {**old, 'internal_name': PROCESS_PAIRS.get(old['internal_name']), 'parameter_name': 'CLspec/[Enzyme]'}
        normalized = km_umol_per_l(branch['km_source']['Value'], branch['km_source'].get('Unit'))
        if (old['parameter_name'] != 'kcat' or expected['internal_name'] is None or identity != expected
                or branch['to_identity'] != expected or normalized != branch['km_umol_per_l']
                or not _positive(branch['kcat_anchor'])):
            raise ValueError('MM-to-linear identity, Km or kcat source drift')
        anchor = branch['kcat_anchor'] / normalized
        multiplier = anchor if shared else 1. / normalized
        path = reviewed_process_parameter_path(compound['Name'], identity)
        if (path in seen or mapping.get('runtime_path') != path or mapping.get('native_unit') != NATIVE_UNIT
                or mapping.get('multiplier') != multiplier or branch['linear_anchor'] != anchor
                or shared and spec['source_anchor_values'][index] != anchor):
            raise ValueError('MM-to-linear runtime mapping or branch anchor drift')
        seen.add(path)
        matches = [p for p in compound['Processes'] if _identity(p, 'CLspec/[Enzyme]') == identity]
        if len(matches) != 1 or matches[0].get('Species') != 'Human':
            raise ValueError('Linear capacity must bind one exact Human process')
        parameters = [p for p in matches[0]['Parameters'] if p['Name'] == 'CLspec/[Enzyme]']
        if len(parameters) != 1 or parameters[0].get('Unit') != NATIVE_UNIT or parameters[0]['Value'] != anchor:
            raise ValueError('Linear source anchor has changed; do not cumulatively apply factors')
        native_value = value * multiplier
        if not math.isfinite(native_value):
            raise ValueError('Linear parameter conversion overflow')
        updates.append((parameters[0], native_value))
    for parameter, native_value in updates:
        parameter['Value'] = native_value
