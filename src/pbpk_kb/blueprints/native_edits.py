"""Source-bound constitutive edits materialized as new standard-human snapshots.

This adapter changes native calculation methods, not anatomy or drug values.
It does not load a PKML, execute a solver, fit parameters, or inspect observations.
Native compilation/execution remains a separate, explicitly budgeted operation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from pbpk_kb.blueprints.design import design_blueprint
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.human_model import load_standard_human_model
from pbpk_kb.blueprints.human_model_snapshot import (
    _ENZYME_PROCESS_PARAMETERS, _MM_PROCESS_PARAMETERS, _enzyme_active_parameter,
)
from pbpk_kb.blueprints.schemas import ProvenanceRef


ADAPTER_VERSION = 'native-constitutive-edits-1.4.0-mechanism-development'
PARTITION_ACTIONS = {
    'partition_pksim_standard': 'Cellular partition coefficient method - PK-Sim Standard',
    'partition_rodgers_rowland': 'Cellular partition coefficient method - Rodgers and Rowland',
}


def reviewed_process_parameter_path(compound_name: str, native_process: dict) -> str:
    """Derive a path for one reviewed process input, never accept arbitrary paths.

    Renal/GFR paths follow native v12 passive-transport naming (also checked in
    this project's freshly generated Amikacin PKML). Biliary fitting is excluded
    because it needs an audited mapping to every duplicated local transport.
    """
    keys = {'internal_name', 'molecule', 'data_source', 'parameter_name'}
    if not isinstance(native_process, dict) or set(native_process) != keys:
        raise ValueError('native_process must provide exactly the four typed identity fields')
    internal, molecule, data_source, parameter = (native_process[k] for k in
                                                ('internal_name', 'molecule', 'data_source', 'parameter_name'))
    if not isinstance(internal, str) or not isinstance(parameter, str):
        raise ValueError('Process type and parameter name must be explicit strings')
    for name in (compound_name, data_source):
        if (not isinstance(name, str) or not name or name != name.strip()
                or any(char in name for char in '|/\\\n\r\t')):
            raise ValueError('Compound and process identities must be plain native names')
    if internal in _ENZYME_PROCESS_PARAMETERS or internal in _MM_PROCESS_PARAMETERS:
        # Native PartialProcess.RefreshName and ProcessToProcessBuilderMapper
        # bind global reaction inputs as Compound-Enzyme-DataSource|Parameter.
        # MM extension is deliberately limited to the downstream kcat input;
        # Km or an upstream Vmax cannot be inferred from an arbitrary path.
        parameters = (_ENZYME_PROCESS_PARAMETERS[internal] if internal in _ENZYME_PROCESS_PARAMETERS
                      else {'kcat'} if 'kcat' in _MM_PROCESS_PARAMETERS[internal] else set())
        if (not isinstance(molecule, str) or not molecule or molecule != molecule.strip()
                or any(char in molecule for char in '|/\\\n\r\t')
                or parameter not in parameters):
            raise ValueError('Unsupported enzyme parameter identity')
        return compound_name + '-' + molecule + '-' + data_source + '|' + parameter
    if molecule is not None:
        raise ValueError('A systemic process must not carry an enzyme identity')
    if internal == 'LiverClearance' and parameter in {'Plasma clearance', 'Specific clearance'}:
        return compound_name + '-Total Hepatic Clearance-' + data_source + '|' + parameter
    names = {'KidneyClearance': ('Renal Clearances', {'Plasma clearance', 'Specific clearance'}),
             'GlomerularFiltration': ('Glomerular Filtration', {'GFR fraction'})}
    if internal in names and parameter in names[internal][1]:
        container = names[internal][0] + '-' + data_source + '-' + compound_name
        return 'Neighborhoods|Kidney_pls_Kidney_ur|' + compound_name + '|' + container + '|' + parameter
    raise ValueError('This process parameter has no reviewed single-parameter runtime mapping')


def _checksum(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _method_source(project_root: Path, method: str) -> tuple[dict, dict]:
    definition = load_standard_human_model(project_root)
    resources = [entry for entry in definition['resources'] if entry['role'] == 'database']
    if len(resources) != 1:
        raise ValueError('One source-verified native database is required')
    resource = resources[0]
    database = (project_root / resource['path']).resolve(strict=True)
    if not database.is_relative_to(project_root):
        raise ValueError('Native method resource leaves the project')
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        rows = connection.execute(
            'SELECT calculation_method, category FROM tab_calculation_methods WHERE calculation_method = ?',
            (method,),
        ).fetchall()
    if rows != [(method, 'DistributionCellular')]:
        raise ValueError('Partition method is absent from the locked DistributionCellular category')
    location = "tab_calculation_methods[calculation_method='" + method + "']; category=DistributionCellular"
    source = {
        'provenance_id': str(uuid5(NAMESPACE_URL, resource['path'] + '#' + location)),
        'provenance_kind': 'model_asset', 'source_id': resource['path'],
        'source_location': location, 'source_checksum': resource['sha256'],
    }
    return source, definition


def _path_differences(before: Any, after: Any, path: str = '') -> list[dict]:
    if isinstance(before, dict) and isinstance(after, dict) and before.keys() == after.keys():
        result = []
        for key in sorted(before):
            escaped = key.replace('~', '~0').replace('/', '~1')
            result.extend(_path_differences(before[key], after[key], path + '/' + escaped))
        return result
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [difference for index, (old, new) in enumerate(zip(before, after, strict=True))
                for difference in _path_differences(old, new, path + '/' + str(index))]
    if _checksum(before) == _checksum(after):
        return []
    return [{'path': path or '/', 'before': deepcopy(before), 'after': deepcopy(after)}]


def _snapshot_from_design(request: BlueprintDesignRequest, root: Path) -> tuple[dict, dict]:
    design = design_blueprint(request, project_root=root)
    binding = design['human_model_binding']
    if (not design['candidates'] or design['candidates'][0]['status'] != 'draft_assembled'
            or not binding.get('instance') or not binding['instance'].get('snapshot')):
        raise ValueError('Native edit requires a complete standard-human base instance')
    return design, binding['instance']['snapshot']


def compile_candidate(
    request: BlueprintDesignRequest | dict, action: str, *, project_root: Path,
    parameter_specs: list[dict] | None = None,
) -> dict[str, Any]:
    """Materialize one reviewed formula choice; native compilation is not run.

    ``parameter_specs`` are unchanged caller metadata, not newly inferred values
    or bounds. The adapter never converts CYP1A2 clearance into a systemic CL:
    those quantities require distinct, explicitly sourced parameterizations.
    """
    if not isinstance(action, str) or action not in PARTITION_ACTIONS:
        raise ValueError('Unsupported native action; only the two registered partition choices are available')
    root = Path(project_root).resolve(strict=True)
    payload = request.model_dump(mode='json') if isinstance(request, BlueprintDesignRequest) else deepcopy(request)
    validated = BlueprintDesignRequest.model_validate(payload)
    if validated.instance is None or validated.human_model is None:
        raise ValueError('Native edits require an explicit registered human model and drug instance')
    if validated.modules:
        raise ValueError('Unapplied generic modules cannot be mixed into a native constitutive candidate')
    if parameter_specs is not None and (not isinstance(parameter_specs, list)
                                        or any(not isinstance(item, dict) for item in parameter_specs)):
        raise ValueError('parameter_specs must be an explicit list of unchanged parameter contracts')
    _checksum(parameter_specs)  # Reject non-JSON and non-finite caller metadata.
    before_design, before_snapshot = _snapshot_from_design(validated, root)
    revised = validated.model_dump(mode='json')
    methods = revised['instance']['compound']['CalculationMethods']
    positions = [i for i, method in enumerate(methods) if method in PARTITION_ACTIONS.values()]
    if len(positions) != 1:
        raise ValueError('The base must have exactly one supported native partition method')
    index = positions[0]
    previous, method = methods[index], PARTITION_ACTIONS[action]
    source, definition = _method_source(root, method)
    if validated.human_model.definition_checksum != definition['checksum']:
        raise ValueError('Candidate source definition differs from the request human-model lock')
    methods[index] = method
    old_source = deepcopy(revised['instance']['compound_sources']['CalculationMethods'])
    revised['instance']['compound_sources']['CalculationMethods'] = source
    design, snapshot = _snapshot_from_design(BlueprintDesignRequest.model_validate(revised), root)
    difference = _path_differences(before_snapshot, snapshot)
    permitted = {f'/Compounds/0/CalculationMethods/{index}',
                 f'/Simulations/0/Compounds/0/CalculationMethods/{index}'}
    if any(item['path'] not in permitted for item in difference):
        raise ValueError('Constitutive edit unexpectedly changed anatomy, drug parameters or administration')
    changed = method != previous
    if {item['path'] for item in difference} != (permitted if changed else set()):
        raise ValueError('The requested constitutive choice was not bound in both native snapshot locations')
    compound = revised['instance']['compound']
    semantics = {
        'base_definition_checksum': definition['checksum'], 'standard_model': '4Comp|Human',
        'partition_method': method,
        'processes': [{key: process[key] for key in ('InternalName', 'Molecule', 'Species', 'DataSource') if key in process}
                      for process in compound['Processes']],
        'compound': compound['Name'], 'change_class': 'constitutive_method',
    }
    parameterization = {
        'compound': {key: value for key, value in compound.items() if key not in {'CalculationMethods', 'Description'}},
        'expression_profiles': revised['instance']['expression_profiles'],
        'parameter_specs': parameter_specs,
    }
    manifest = {
        'adapter_version': ADAPTER_VERSION, 'action': action, 'change_class': 'constitutive_method',
        'topology_changed': False, 'drug_parameter_values_changed': False, 'human_core_preserved': True,
        'from_method': previous, 'to_method': method, 'identity_action': not changed,
        'source': source, 'source_binding_verified': True, 'parent_method_source': old_source,
        'source_verification_scope': 'registered native calculation-method identity and locked database; not scientific suitability',
        'pathdiff': difference, 'parent_snapshot_sha256': _checksum(before_snapshot),
        'snapshot_sha256': _checksum(snapshot), 'compile_status': 'not_run', 'simulation_status': 'not_run',
        'scientific_status': 'not_evaluated', 'execution': {'solver_calls': 0, 'gpu_jobs': 0, 'llm_calls': 0},
        'limitations': ['Constitutive-method comparison, not a novel organ topology.',
                        'Distinct formulas do not establish clinically meaningful or independent predictive improvement.'],
    }
    return {
        'structure_id': action, 'request': revised, 'design': design, 'snapshot': snapshot,
        'semantic_hash': _checksum(semantics), 'parameterization_hash': _checksum(parameterization),
        'edit_count': int(changed), 'edit_manifest': manifest, 'parameter_specs': deepcopy(parameter_specs),
    }


def partition_candidates(
    request: BlueprintDesignRequest | dict, *, project_root: Path,
    parameter_specs: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Return the same two legal formula choices for every comparison method."""
    return [compile_candidate(request, action, project_root=project_root, parameter_specs=parameter_specs)
            for action in PARTITION_ACTIONS]


def _require_source_human_enzyme(instance: dict, process: dict) -> None:
    """Bind a partial-process DTO to its explicit Human biological context.

    PK-Sim CompoundProcessMapper omits Species on non-species-dependent partial
    processes and leaves template species unchanged when that field is absent.
    Keep that DTO untouched. An explicit conflicting Species, including null,
    never inherits the individual's sourced Human enzyme context.
    """
    if process.get('Species', 'Human') != 'Human':
        raise ValueError('Single MM process explicitly conflicts with Human species')
    individual = instance.get('individual', {})
    if individual.get('OriginData', {}).get('Species') != 'Human':
        raise ValueError('Single MM capacity requires an explicitly Human individual')
    profiles = [p for p in instance.get('expression_profiles', [])
                if p.get('Molecule') == process.get('Molecule')]
    if len(profiles) != 1 or profiles[0].get('Type') != 'Enzyme' or profiles[0].get('Species') != 'Human':
        raise ValueError('Single MM capacity needs exactly one sourced Human enzyme profile')
    profile = profiles[0]
    identifier = profile['Molecule'] + '|Human|' + profile['Category']
    if individual.get('ExpressionProfiles', []).count(identifier) != 1:
        raise ValueError('Human enzyme profile must be selected exactly once by the individual')
    ProvenanceRef.model_validate(instance.get('human_source'))
    ProvenanceRef.model_validate(instance.get('expression_profile_sources', {}).get(identifier))


def apply_parameter_values(
    request: BlueprintDesignRequest | dict, parameter_specs: list[dict], values: list[float],
    provenance: dict,
) -> dict[str, Any]:
    """Write at most two reviewed parameter mappings into explicit drug input.

    Values and bounds are supplied by the audited calibration caller. Fixed CYP
    branch fractions must sum to one; permeability is converted dm/min -> cm/s.
    A shared MM factor multiplies each separately sourced kcat anchor, preserving
    branch ratios and Km. It must be applied to the original anchored instance,
    never cumulatively to an already scaled instance.
    No unrelated drug, anatomy, protocol or constitutive method is changed.
    """
    payload = request.model_dump(mode='json') if isinstance(request, BlueprintDesignRequest) else deepcopy(request)
    result = BlueprintDesignRequest.model_validate(payload).model_dump(mode='json')
    if result['instance'] is None:
        raise ValueError('Parameter application requires an explicit drug instance')
    source = ProvenanceRef.model_validate(provenance).model_dump(mode='json')
    if (not isinstance(parameter_specs, list) or not isinstance(values, list)
            or len(parameter_specs) != len(values) or len(parameter_specs) > 2):
        raise ValueError('Provide matching lists of at most two reviewed parameter mappings')
    ids = [spec['parameter_id'] for spec in parameter_specs]
    if len(ids) != len(set(ids)):
        raise ValueError('Parameter mappings must have unique identifiers')

    def finite(value):
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)

    compound = result['instance']['compound']
    for spec, value in zip(parameter_specs, values, strict=True):
        lower, upper = spec['lower_bound'], spec['upper_bound']
        if not all(finite(x) for x in (value, lower, upper)) or not lower <= value <= upper:
            raise ValueError('Estimated parameter exceeds finite source-provided bounds')
        mappings = spec['runtime_mappings']
        if not isinstance(mappings, list) or not mappings:
            raise ValueError('An explicit native runtime mapping is required')
        if spec['parameter_id'] == 'total_cyp1a2_specific_clearance':
            if spec['unit'] != '1/min' or value < 0:
                raise ValueError('CYP1A2 specific clearance must be nonnegative in 1/min')
            multipliers = [mapping['multiplier'] for mapping in mappings]
            if (any(not finite(x) or x < 0 for x in multipliers)
                    or not math.isclose(sum(multipliers), 1., rel_tol=0., abs_tol=1e-12)):
                raise ValueError('CYP1A2 fixed branch fractions must be nonnegative and sum to one')
            covered = set()
            for mapping in mappings:
                prefix = compound['Name'] + '-CYP1A2-'
                path = mapping['runtime_path']
                if not isinstance(path, str) or not path.startswith(prefix) or not path.endswith('|Specific clearance'):
                    raise ValueError('Unsupported CYP1A2 native mapping or compound identity')
                process_source = path.removeprefix(prefix).removesuffix('|Specific clearance')
                if process_source in covered:
                    raise ValueError('CYP1A2 branch is mapped more than once')
                matches = [process for process in compound['Processes']
                           if process.get('InternalName') == 'MetabolizationIntrinsic_FirstOrder'
                           and process.get('Molecule') == 'CYP1A2' and process['DataSource'] == process_source]
                if len(matches) != 1:
                    raise ValueError('CYP1A2 process mapping is not unique')
                parameters = [p for p in matches[0]['Parameters'] if p['Name'] == 'Specific clearance']
                if len(parameters) != 1 or parameters[0]['Unit'] != '1/min':
                    raise ValueError('Native specific clearance unit differs from the calibrated value')
                parameters[0]['Value'] = value * mapping['multiplier']
                covered.add(process_source)
            all_branches = {process['DataSource'] for process in compound['Processes']
                            if process.get('Molecule') == 'CYP1A2'}
            if covered != all_branches:
                raise ValueError('Total CYP1A2 clearance must cover every selected CYP1A2 branch')
            result['instance']['compound_sources']['Processes'] = source
        elif spec['parameter_id'] == 'specific_intestinal_permeability_transcellular':
            expected_path = compound['Name'] + '|Specific intestinal permeability (transcellular)'
            if (spec['unit'] != 'dm/min' or value <= 0 or len(mappings) != 1
                    or not isinstance(mappings[0], dict)
                    or mappings[0].get('runtime_path') != expected_path
                    or not finite(mappings[0].get('multiplier'))
                    or mappings[0]['multiplier'] != 1.0):
                raise ValueError('Permeability requires the exact positive dm/min native mapping')
            alternatives = compound.get('IntestinalPermeability', [])
            if len(alternatives) != 1:
                raise ValueError('One explicit intestinal permeability alternative is required')
            parameters = alternatives[0]['Parameters']
            if (len(parameters) != 1 or parameters[0]['Name'] != 'Specific intestinal permeability (transcellular)'
                    or parameters[0]['Unit'] != 'cm/s'):
                raise ValueError('Native permeability mapping drift')
            parameters[0]['Value'] = value / 6.0
            result['instance']['compound_sources']['IntestinalPermeability'] = source
        elif spec['parameter_id'] in {'shared_process_parameter', 'single_process_kcat'} and spec.get('mechanism_conversion'):
            from pbpk_kb.blueprints.mechanism_edits import apply_linearized_parameter
            apply_linearized_parameter(compound, spec, value)
            result['instance']['compound_sources']['Processes'] = source
        elif spec['parameter_id'] == 'single_process_kcat':
            ProvenanceRef.model_validate(spec.get('source'))
            identity = spec.get('native_process')
            expected_path = reviewed_process_parameter_path(compound['Name'], identity)
            if (identity['internal_name'] not in _MM_PROCESS_PARAMETERS or identity['parameter_name'] != 'kcat'
                    or spec['unit'] != '1/min' or lower < 0 or value < 0
                    or len(mappings) != 1 or mappings[0].get('runtime_path') != expected_path
                    or mappings[0].get('native_unit', spec['unit']) != '1/min'
                    or mappings[0].get('multiplier') != 1.0):
                raise ValueError('Single MM kcat requires one exact nonnegative 1/min identity mapping')
            matches = [p for p in compound['Processes'] if
                       (p.get('InternalName'), p.get('Molecule'), p.get('DataSource')) ==
                       (identity['internal_name'], identity['molecule'], identity['data_source'])]
            if len(matches) != 1:
                raise ValueError('Single MM kcat must uniquely bind a source Human process')
            _require_source_human_enzyme(result['instance'], matches[0])
            kcat = [p for p in matches[0]['Parameters'] if p['Name'] == 'kcat']
            km = [p for p in matches[0]['Parameters'] if p['Name'] == 'Km']
            if (len(kcat) != 1 or kcat[0].get('Unit') != '1/min' or len(km) != 1
                    or km[0].get('Unit') not in _MM_PROCESS_PARAMETERS[identity['internal_name']]['Km']
                    or not finite(km[0].get('Value')) or km[0]['Value'] <= 0):
                raise ValueError('Single MM capacity needs a native kcat and sourced positive Km')
            kcat[0]['Value'] = value
            result['instance']['compound_sources']['Processes'] = source
        elif spec['parameter_id'] == 'shared_process_parameter':
            ProvenanceRef.model_validate(spec.get('source'))
            identities, anchors = spec.get('native_processes'), spec.get('source_anchor_values')
            if (spec.get('parameterization') != 'shared_factor' or spec.get('unit') != 'dimensionless'
                    or lower < 0 or value < 0 or not isinstance(identities, list)
                    or not 2 <= len(identities) <= 8 or not isinstance(anchors, list)
                    or len(identities) != len(mappings) or len(anchors) != len(identities)):
                raise ValueError('Shared MM capacity requires a nonnegative dimensionless factor and explicit branch anchors')
            seen, enzyme, updates = set(), None, []
            for identity, anchor, mapping in zip(identities, anchors, mappings, strict=True):
                expected_path = reviewed_process_parameter_path(compound['Name'], identity)
                internal = identity['internal_name']
                if (internal not in _MM_PROCESS_PARAMETERS or identity['parameter_name'] != 'kcat'
                        or not finite(anchor) or anchor <= 0 or not isinstance(mapping, dict)
                        or mapping.get('runtime_path') != expected_path or mapping.get('native_unit') != '1/min'
                        or not finite(mapping.get('multiplier')) or mapping['multiplier'] != anchor):
                    raise ValueError('Shared MM capacity requires exact reviewed kcat paths, units and source multipliers')
                branch = (internal, identity['molecule'], identity['data_source'])
                if branch in seen or (enzyme is not None and enzyme != identity['molecule']):
                    raise ValueError('Shared MM factor must bind distinct branches of one explicitly sourced enzyme')
                seen.add(branch); enzyme = identity['molecule']
                matches = [process for process in compound['Processes']
                           if (process.get('InternalName'), process.get('Molecule'), process.get('DataSource')) == branch]
                if len(matches) != 1 or matches[0].get('Species', 'Human') != 'Human':
                    raise ValueError('Each shared MM branch must uniquely bind a source Human process')
                parameters = matches[0]['Parameters']
                kcat = [p for p in parameters if p.get('Name') == 'kcat']
                km = [p for p in parameters if p.get('Name') == 'Km']
                if (len(kcat) != 1 or kcat[0].get('Unit') != '1/min'
                        or not finite(kcat[0].get('Value')) or kcat[0]['Value'] != anchor
                        or len(km) != 1 or km[0].get('Unit') not in _MM_PROCESS_PARAMETERS[internal]['Km']
                        or not finite(km[0].get('Value')) or km[0]['Value'] <= 0):
                    raise ValueError('Shared MM capacity needs unchanged original kcat anchors and a sourced positive Km')
                scaled = value * anchor
                if not finite(scaled):
                    raise ValueError('Shared MM factor would overflow native kcat')
                updates.append((kcat[0], scaled))
            for parameter, scaled in updates:
                parameter['Value'] = scaled
            result['instance']['compound_sources']['Processes'] = source
        elif spec['parameter_id'] == 'single_process_clearance':
            ProvenanceRef.model_validate(spec.get('source'))
            identity = spec.get('native_process')
            expected_path = reviewed_process_parameter_path(compound['Name'], identity)
            if (len(mappings) != 1 or not isinstance(mappings[0], dict)
                    or mappings[0].get('runtime_path') != expected_path
                    or not finite(mappings[0].get('multiplier')) or mappings[0]['multiplier'] != 1.0):
                raise ValueError('Single process clearance requires its exact reviewed identity mapping')
            matches = [process for process in compound['Processes']
                       if process.get('InternalName') == identity['internal_name']
                       and process.get('Molecule') == identity['molecule']
                       and process.get('DataSource') == identity['data_source']]
            if len(matches) != 1 or matches[0].get('Species') != 'Human':
                raise ValueError('The sourced human process must be uniquely bound')
            process = matches[0]
            parameters = [p for p in process['Parameters'] if p.get('Name') == identity['parameter_name']]
            if len(parameters) != 1 or lower < 0 or value < 0:
                raise ValueError('One nonnegative source-backed clearance input is required')
            parameter = parameters[0]
            names = {p['Name'] for p in process['Parameters']}
            if identity['internal_name'] in _ENZYME_PROCESS_PARAMETERS:
                active = _enzyme_active_parameter(identity['internal_name'], names)
                allowed_units = _ENZYME_PROCESS_PARAMETERS[identity['internal_name']][identity['parameter_name']]
            elif identity['internal_name'] == 'GlomerularFiltration':
                active, allowed_units = 'GFR fraction', {None, ''}
                if upper > 1 or spec['unit'] not in {'dimensionless', 'fraction', '1'}:
                    raise ValueError('GFR fraction bounds and units must be dimensionless in [0, 1]')
            else:
                active = 'Specific clearance' if 'Specific clearance' in names else 'Plasma clearance'
                allowed_units = {'1/min', '1/h'} if active == 'Specific clearance' else {'ml/min/kg', 'l/h/kg', 'ml/h/kg'}
            if active != identity['parameter_name']:
                raise ValueError('Cannot fit an upstream input hidden by an explicitly supplied downstream clearance')
            if parameter.get('Unit') not in allowed_units:
                raise ValueError('Native clearance input unit is not in the reviewed dimension')
            if active != 'GFR fraction' and spec['unit'] != parameter.get('Unit'):
                raise ValueError('Clearance specification must use the explicit source native unit')
            parameter['Value'] = value
            result['instance']['compound_sources']['Processes'] = source
        elif spec['parameter_id'] in {'cyp3a4_intrinsic_clearance', 'lipophilicity'}:
            # These contracts must carry the separately audited input source
            # for their start/bounds; `source` above records this application.
            ProvenanceRef.model_validate(spec.get('source'))
            if (len(mappings) != 1 or not isinstance(mappings[0], dict)
                    or not finite(mappings[0].get('multiplier')) or mappings[0]['multiplier'] != 1.0):
                raise ValueError('Reviewed scalar parameters require one exact identity mapping')
            if spec['parameter_id'] == 'cyp3a4_intrinsic_clearance':
                if spec['unit'] != 'l/min' or value < 0 or lower < 0:
                    raise ValueError('CYP3A4 intrinsic clearance must be nonnegative in l/min')
                matches = [process for process in compound['Processes'] if process.get('Molecule') == 'CYP3A4']
                if (len(matches) != 1 or matches[0].get('InternalName') != 'MetabolizationIntrinsic_FirstOrder'
                        or matches[0].get('Species') != 'Human'):
                    raise ValueError('One explicit human first-order CYP3A4 process is required')
                process = matches[0]
                expected_path = compound['Name'] + '-CYP3A4-' + process['DataSource'] + '|Intrinsic clearance'
                parameters = process['Parameters']
                if (mappings[0].get('runtime_path') != expected_path or len(parameters) != 1
                        or parameters[0].get('Name') != 'Intrinsic clearance' or parameters[0].get('Unit') != 'l/min'):
                    raise ValueError('CYP3A4 intrinsic clearance native mapping or unit drift')
                parameters[0]['Value'] = value
                result['instance']['compound_sources']['Processes'] = source
            else:
                alternatives = compound.get('Lipophilicity', [])
                if (spec['unit'] != 'Log Units' or len(alternatives) != 1
                        or mappings[0].get('runtime_path') != compound['Name'] + '|Lipophilicity'):
                    raise ValueError('Lipophilicity requires one explicit native Log Units alternative')
                parameters = alternatives[0]['Parameters']
                if (len(parameters) != 1 or parameters[0].get('Name') != 'Lipophilicity'
                        or parameters[0].get('Unit') != 'Log Units'):
                    raise ValueError('Native lipophilicity mapping or unit drift')
                parameters[0]['Value'] = value
                result['instance']['compound_sources']['Lipophilicity'] = source
        else:
            raise ValueError('No reviewed native snapshot adapter for parameter ' + spec['parameter_id'])
    return result
