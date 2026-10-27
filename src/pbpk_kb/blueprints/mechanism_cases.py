"""Prepare source-locked joint mechanism/parameter cases, preserving old runs."""
from copy import deepcopy
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .experiment_runtime import candidate_space, file_sha, save, validate_case
from .native_edits import PARTITION_ACTIONS
from .mechanism_edits import ACTION


INPUTS = {
    'alprazolam': 'experiments/blueprint_design/development_inputs/alprazolam_reconstruction_20260920_v2/input_package.json',
    'midazolam': 'experiments/blueprint_design/development_inputs/midazolam_reconstruction_20260920/input_package.json',
}


def prepare_mechanism_cases(project_root, output):
    root = Path(project_root).resolve(strict=True)
    output = (root / output).resolve()
    if output == root or not output.is_relative_to(root) or output.exists():
        raise ValueError('Use a new project-local directory for mechanism cases')
    records = []
    for drug, source in INPUTS.items():
        path = root / source
        bundle = json.loads(path.read_text())
        case = bundle['case']
        parent_id = case['case_id']
        case['case_id'] = parent_id + '_kinetics_v1'
        if drug == 'midazolam':
            from .midazolam_cases import SOURCE, PI_NAME
            source_path = root / SOURCE[0]
            if file_sha(source_path) != SOURCE[1]:
                raise ValueError('Midazolam parameter source checksum drift')
            snapshot = json.loads(source_path.read_text())
            pi = next(p for p in snapshot['ParameterIdentifications'] if p['Name'] == PI_NAME)
            entry = next(p for p in pi['IdentificationParameters'] if p['Name'] == 'kcat CYP3A4')
            parameters = {p['Name']: p for p in entry['Parameters']}
            names = ('Start value', 'MinValue', 'MaxValue')
            values = [parameters[name]['Value'] for name in names]
            if (values != [7.8703703704, 1., 20.] or entry.get('UseAsFactor', False)
                    or entry['Scaling'] != 'Log'
                    or any(parameters[name]['Unit'] != '1/min' for name in names)
                    or len(entry['LinkedParameters']) != 4
                    or any(not p.endswith('|Midazolam-CYP3A4-Optimized|kcat') for p in entry['LinkedParameters'])):
                raise ValueError('Midazolam source kcat PI contract drift')
            ref = {'provenance_id':str(uuid5(NAMESPACE_URL, SOURCE[0] + PI_NAME + 'kcat CYP3A4')),
                'provenance_kind':'model_asset','source_id':SOURCE[0], 'source_checksum':SOURCE[1],
                'source_location':f'ParameterIdentifications[{PI_NAME}].kcat CYP3A4; source start and numerical bounds'}
            spec = {'parameter_id':'single_process_kcat', 'initial_value':values[0], 'lower_bound':values[1],
                'upper_bound':values[2], 'unit':'1/min', 'search_scale':'log', 'sensitivity_scale':'log',
                'native_process':{'internal_name':'MetabolizationLiverMicrosomes_MM','molecule':'CYP3A4',
                                  'data_source':'Optimized','parameter_name':'kcat'},
                'runtime_mappings':[{'runtime_path':'Midazolam-CYP3A4-Optimized|kcat','multiplier':1.}],
                'source':ref, 'bound_semantics':'historical_numerical_search_guards',
                'anchor_may_be_historically_fitted':True}
            case['parameter_search']['parameters'].insert(0, spec)
            case['parameter_search'].pop('fixed_mm_capacity_reason', None)
            requests = [case['design_request']] + [s['design_request'] for s in case['scenarios'] + bundle['external_scenarios']]
            for request in requests:
                compound = request['instance']['compound']
                matches = [p for p in compound['Processes'] if p.get('InternalName')=='MetabolizationLiverMicrosomes_MM'
                           and p.get('Molecule')=='CYP3A4' and p['DataSource']=='Optimized']
                if len(matches) != 1:
                    raise ValueError('Expected one explicit source CYP3A4 MM branch')
                next(p for p in matches[0]['Parameters'] if p['Name']=='kcat')['Value'] = spec['initial_value']
                compound['Description'] = 'Historical reconstruction: source-PI CYP3A4 kcat and lipophilicity fitted; UGT1A4 source anchor fixed; source human core unchanged.'
                request['instance']['compound_sources']['Processes'] = deepcopy(ref)
        case['mechanism_search'] = {
            'version':'native-metabolism-search-v1',
            'expected_structure_ids':list(PARTITION_ACTIONS) + [a+'__'+ACTION for a in PARTITION_ACTIONS],
            'changed_mechanism':'native saturable enzyme kinetics versus source-matched first-order approximation',
            'human_base_preserved':True, 'anatomical_topology_search':False,
            'candidate_approximation_is_not_clinically_prevalidated':True,
            'source_initialization_may_be_historically_fitted':True,
            'guidance_policy':'No fabricated differential requirement/evidence score; equal guidance unless separately source-qualified.',
            'parent_case_id':parent_id,'parent_bundle_path':source,'parent_bundle_sha256':file_sha(path),
        }
        case.setdefault('limitations', []).extend([
            'These studies remain historical reconstruction, not independent drug-level validation.',
            'Mechanism search changes native enzyme rate laws; it does not search new anatomy.',
            'Low-concentration first-order candidates retain approximation uncertainty; plasma alone cannot certify local substrate/Km.',
        ])
        validate_case(case)
        candidates = candidate_space(case, root=root)
        if len(candidates) != 16:
            raise ValueError('Expected four native structures times four parameter subsets')
        target = output / drug / 'input_package.json'
        save(target, bundle)
        save(target.parent / 'candidate_space.json', candidates)
        records.append({'compound':drug, 'case_id':case['case_id'], 'bundle_path':str(target.relative_to(root)),
            'bundle_sha256':file_sha(target), 'candidate_count':len(candidates), 'native_structure_count':4,
            'tier':case['tier'], 'development_study_count':len({s['study_group_id'] for s in case['scenarios']}),
            'external_study_count':len({s['study_group_id'] for s in bundle['external_scenarios']})})
    manifest = {'schema_version':'native-mechanism-cases-v1','cases':records,
                'independent_main_benchmark':False, 'sealed_external_observations_decoded':False}
    save(output / 'manifest.json', manifest)
    return manifest
