"""Furosemide historical fitting inputs and a separately blinded new study.

Uses actual source protocols, individuals, transporter directions and release
parameters. Historical calibration is not represented as independent evidence.
"""
from copy import deepcopy
import json
from pathlib import Path

from .cohort_cases import _normalise_protocol, _output_schema, _slug, qualify_sampled_endpoints
from .development_case import _one, _plain, _reference
from .multidrug_cases import _external_observations, _metadata
from .experiment_runtime import candidate_space, file_sha, save, validate_case
from .native_edits import PARTITION_ACTIONS, reviewed_process_parameter_path
from .mechanism_edits import ACTION
from .external_source_isolation import exclusion_policy, audit_design_inputs

SOURCE = 'artifacts/pbpk-model-assets/public_collection_20260909/Furosemide-Model__bd0c455aa3c8/source/Furosemide-Model.json'
PROTOCOL = ('experiments/blueprint_design/independent_cohort_20260920/protocol_revision_20260920/'
            'prediction_metadata/furosemide_pmc8416862_trial2_baseline.json')
DESIGN = 'runtime/blueprint_mechanism_worktree_20260920/native_verification/furosemide_source_40mg_tablet/design_request.json'
STUDIES = {
    'Haegeli 2007': ('calibration', 'doi:10.1111/j.1365-2125.2007.03035.x'),
    'Waller 1982': ('selection', 'doi:10.1002/jps.2600711006'),
}
SCENES = [
    ('Haegeli 2007','i.v. 20 mg (bolus)_Haegeli 2007_33+34','Haegeli 2007_20_i.v._33'),
    ('Haegeli 2007','p.o., 20 mg (tab)_Haegeli 2007_35+36','Haegeli 2007_20_p.o._35'),
    ('Waller 1982','i.v. 40 mg (2 min)_Waller 1982_97','Waller 1982_40_i.v._97'),
    ('Waller 1982','p.o., 40 mg (sol)_Waller 1982_99','Waller 1982_40_p.o._99'),
]


def prepare_furosemide_case(project_root, output):
    root = Path(project_root).resolve(strict=True)
    output = (root/output).resolve()
    if output == root or not output.is_relative_to(root) or output.exists():
        raise ValueError('Prepare Furosemide in a new project-local directory')
    template = json.loads((root/DESIGN).read_text())
    snapshot = json.loads((root/SOURCE).read_text())
    checksum = file_sha(root/SOURCE)
    if template['instance']['compound_sources']['Processes']['source_checksum'] != checksum:
        raise ValueError('Source differs from the engineering-verified Furosemide instance')
    ref = lambda location: _reference(SOURCE, checksum, location)
    compound = template['instance']['compound']
    mm = _one([p for p in compound['Processes'] if p.get('Molecule')=='UGT1A9'], 'InternalName','MetabolizationSpecific_MM')
    kcat = _one(mm['Parameters'],'Name','kcat')
    logp = _one(compound['Lipophilicity'][0]['Parameters'],'Name','Lipophilicity')
    if kcat['Unit'] != '1/min' or kcat['Value']<=0 or logp['Unit'] != 'Log Units':
        raise ValueError('Source capacity/lipophilicity units changed')
    identity = {'internal_name':mm['InternalName'],'molecule':mm['Molecule'],
                'data_source':mm['DataSource'],'parameter_name':'kcat'}
    specs = [
        {'parameter_id':'single_process_kcat','initial_value':kcat['Value'],
            'lower_bound':kcat['Value']*.1,'upper_bound':kcat['Value']*10,
            'unit':'1/min','search_scale':'log','sensitivity_scale':'log','native_process':identity,
            'runtime_mappings':[{'runtime_path':reviewed_process_parameter_path('Furosemide',identity),'multiplier':1.}],
            'source':ref('Compounds[Furosemide].Processes[UGT1A9-Kerdpin2008].kcat; historical fitted anchor'),
            'bound_semantics':'registered historical-source multiplier guards 0.1..10; not biological uncertainty',
            'anchor_may_be_historically_fitted':True},
        {'parameter_id':'lipophilicity','initial_value':logp['Value'],
            'lower_bound':logp['Value']-1,'upper_bound':logp['Value']+1,
            'unit':'Log Units','search_scale':'linear','sensitivity_scale':'bounded_linear',
            'runtime_mappings':[{'runtime_path':'Furosemide|Lipophilicity','multiplier':1.}],
            'source':ref('Compounds[Furosemide].Lipophilicity; historical fitted anchor'),
            'bound_semantics':'registered historical anchor +/-1 Log Unit numerical guard; not biological uncertainty',
            'anchor_may_be_historically_fitted':True},
    ]
    policy = exclusion_policy([PROTOCOL],project_root=root)
    scenarios = []
    proteins = {p['Molecule'] for p in compound['Processes'] if 'Molecule' in p}
    mw = _one(compound['Parameters'],'Name','Molecular weight')['Value']
    for study,name,observed_name in SCENES:
        simulation = _one(snapshot['Simulations'],'Name',name)
        observed = _one(snapshot['ObservedData'],'Name',observed_name)
        metadata = _metadata(observed)
        if metadata.get('Molecule') != 'Furosemide' or metadata.get('Compartment') != 'Plasma':
            raise ValueError('A urinary/metabolite observation cannot enter plasma fitting')
        if metadata.get('Study Id') != study or metadata.get('Species') != 'Human':
            raise ValueError('Source study identity changed')
        role,group = STUDIES[study]
        identifier = 'furosemide_' + _slug(name)
        selected = simulation['Compounds'][0]
        if selected['Name'] != 'Furosemide':
            raise ValueError('Wrong simulation compound')
        request = deepcopy(template)
        instance = request['instance']
        original = _one(snapshot['Individuals'],'Name',simulation['Individual'])
        individual = {k:_plain(original[k]) for k in ('Name','Seed','OriginData')}
        individual['OriginData'].pop('CalculationMethods',None)
        ids = [i for i in original['ExpressionProfiles'] if i.split('|')[0] in proteins]
        individual['ExpressionProfiles'] = ids
        instance['individual'] = individual
        instance['generated_demographics'] = [n for n in ('Weight','Height') if n not in individual['OriginData']]
        instance['expression_profiles'] = [_plain(next(p for p in snapshot['ExpressionProfiles']
            if p['Molecule']+'|'+p['Species']+'|'+p['Category']==i)) for i in ids]
        instance['expression_profile_sources'] = {i:ref('ExpressionProfiles['+i+']; exact study source individual') for i in ids}
        instance['human_source'] = ref('Individuals['+original['Name']+']; source demographics, not fitted to final study')
        source_protocol = _one(snapshot['Protocols'],'Name',selected['Protocol']['Name'])
        if source_protocol['ApplicationType']=='Oral':
            if source_protocol['DosingInterval']!='Single':
                raise ValueError('This clinical case requires an explicit single oral dose')
            protocol = _plain(source_protocol)
            protocol['Name'] = 'PROTO_' + identifier
            protocol['Parameters'] = [p for p in protocol['Parameters'] if p['Name']!='End time']
            # Native Single protocol ignores repeat End time; source sampling
            # and simulation output window are independently preserved below.
        else:
            protocol = _normalise_protocol(source_protocol,identifier)
        instance['protocol'] = protocol
        instance['protocol_source'] = ref('Protocols['+selected['Protocol']['Name']+']; unchanged source dose/time')
        route = 'oral' if protocol['ApplicationType']=='Oral' else 'intravenous'
        if route == 'oral':
            form = selected['Protocol']['Formulations'][0]['Name']
            instance['formulation'] = _plain(_one(snapshot['Formulations'],'Name',form))
            instance['formulation'].setdefault('Parameters',[])
            instance['formulation_source'] = ref('Formulations['+form+']; exact source release parameters')
        else:
            instance['formulation'] = None
            instance['formulation_source'] = None
        instance['simulation_name'] = identifier
        instance['output_schema'] = _output_schema(simulation)
        dose = _one(protocol['Parameters'],'Name','InputDose')
        request['requirement'].update(route=route,dose=dose['Value'],dose_unit=dose['Unit'],
            formulation=instance['formulation']['Name'] if instance['formulation'] else None)
        request['studies'] = []
        audit_design_inputs({'request':request,'parameters':specs},policy)
        if observed['BaseGrid']['Unit']!='h':
            raise ValueError('Unexpected native observed time unit')
        times = [t*60 for t in observed['BaseGrid']['Values']]
        sampling = qualify_sampled_endpoints(times,source=ref('ObservedData['+observed_name+'].BaseGrid'))
        end = max(_one(interval['Parameters'],'Name','End time')['Value'] for interval in instance['output_schema'])*60
        if max(times)>end:
            raise ValueError('Source output window cannot cover observations; do not truncate')
        observations = _external_observations(observed,mw)
        scenarios.append({'scenario_id':identifier,'compound_id':request['requirement']['compound_semantic_id'],
            'study_group_id':group,'role':role,'tier':'reconstruction','design_request':request,
            'target_output':'Organism|PeripheralVenousBlood|Furosemide|Plasma (Peripheral Venous Blood)',
            'observations':observations,'observation_count':len(observations),
            'source_study_id':study,'source_observation_name':observed_name,**sampling})
    case = {'schema_version':'native-furosemide-case-v1','case_id':'furosemide_haegeli_waller_kinetics_v1',
        'purpose':'historical_development_reconstruction','tier':'reconstruction',
        'compound_id':template['requirement']['compound_semantic_id'],
        'requirement':deepcopy(scenarios[0]['design_request']['requirement']),
        'design_request':deepcopy(scenarios[0]['design_request']),
        'parameter_search':{'parameters':specs,'objective':'study_curve_equal_log_rmse'},'scenarios':scenarios,
        'source_audit':[{'path':SOURCE,'sha256':checksum},{'path':DESIGN,'sha256':file_sha(root/DESIGN)},
                        {'path':PROTOCOL,'sha256':file_sha(root/PROTOCOL)}],
        'forbidden_study_group_ids':[p['study_id'] for p in policy['protocols']],
        'independent_endpoint_protocols':[{'path':PROTOCOL,'sha256':file_sha(root/PROTOCOL)}],
        'mechanism_search':{'version':'native-metabolism-search-v1',
            'expected_structure_ids':list(PARTITION_ACTIONS)+[a+'__'+ACTION for a in PARTITION_ACTIONS]},
        'limitations':['Clinical fitting/selection sources were used by the original public model.',
            'Historical source-fitted UGT and release anchors retained; no independent-prior claim.',
            'All native transport remains fixed; kinetic candidates edit metabolism only.',
            'External2021 is sealed endpoint validation, not a historical-source curve or automatic strict-main qualification.']}
    validate_case(case)
    candidates = candidate_space(case,root=root)
    save(output/'input_package.json',{'case':case,'external_scenarios':[]})
    save(output/'candidate_space.json',candidates)
    save(output/'source_isolation.json',policy)
    report={'case_id':case['case_id'],'compound_id':case['compound_id'],'tier':case['tier'],
        'bundle_path':str((output/'input_package.json').relative_to(root)),
        'bundle_sha256':file_sha(output/'input_package.json'),'candidate_count':len(candidates),
        'study_counts':{role:len({s['study_group_id'] for s in scenarios if s['role']==role}) for role in ('calibration','selection')},
        'curve_count':len(scenarios),'observation_count':sum(len(s['observations']) for s in scenarios),
        'independent_observations_decoded':False}
    save(output/'manifest.json',report)
    return report
