"""Complete route absence cannot hide engineering or parent-identity failures."""
from copy import deepcopy
import json
import pytest
from recommendation_execution_repairs import json_source_absence as j,engineering_gate as g


def document():
    return {'Version':80,'Compounds':[{'Name':'Target'},{'Name':'Partner'}],
      'Protocols':[{'Name':'IV','ApplicationType':'Intravenous'},
                   {'Name':'Oral','Schemas':[{'SchemaItems':[{'ApplicationType':'Oral'}]}]}],
      'Simulations':[{'Name':'monotherapy','Compounds':[{'Name':'Target','Protocol':{'Name':'IV'}}]},
                     {'Name':'DDI','Compounds':[{'Name':'Target','Protocol':{'Name':'IV'}},
                                                {'Name':'Partner','Protocol':{'Name':'Oral'}}]},
                     {'Name':'Other','Compounds':[{'Name':'Partner','Protocol':{'Name':'Oral'}}]}]}


def test_all_simulations_and_all_drugs_inspected_without_ddi_exclusion():
    d=document();facts=j.route_inventory(d,'target')
    assert len(facts)==3
    assert [f['target_routes'] for f in facts]==[['Intravenous'],[],['Intravenous']]
    assert len(facts[0]['all_administrations'])==2
    d['Protocols'][0]['ApplicationType']='IntravenousBolus'
    assert j.route_inventory(d,'target')==facts


@pytest.mark.parametrize('fault',['no_target','no_admin','duplicate_compound','duplicate_protocol','duplicate_simulation','duplicate_selection','missing_protocol','unknown_type','empty_schema','empty_items','custom_event','simulation_event','route_override','mixed_protocol','undeclared_drug'])
def test_any_unresolved_binding_rejects_absence(fault):
    d=document()
    if fault=='no_target':d['Compounds'][0]['Name']='Target_alias'
    elif fault=='no_admin':
        for sim in d['Simulations']:
            for selected in sim['Compounds']:
                if selected['Name']=='Target':selected.pop('Protocol')
    elif fault=='duplicate_compound':d['Compounds'].append(dict(d['Compounds'][0]))
    elif fault=='duplicate_protocol':d['Protocols'].append(dict(d['Protocols'][0]))
    elif fault=='duplicate_simulation':d['Simulations'].append(dict(d['Simulations'][0]))
    elif fault=='duplicate_selection':d['Simulations'][0]['Compounds']*=2
    elif fault=='missing_protocol':d['Simulations'][0]['Compounds'][0]['Protocol']['Name']='missing'
    elif fault=='unknown_type':d['Protocols'][0]['ApplicationType']='Novel'
    elif fault=='empty_schema':d['Protocols'][1]['Schemas']=[]
    elif fault=='empty_items':d['Protocols'][1]['Schemas'][0]['SchemaItems']=[]
    elif fault=='custom_event':d['Events']=[{'Name':'Custom'}]
    elif fault=='simulation_event':d['Simulations'][0]['Events']=[{'Name':'Custom'}]
    elif fault=='route_override':d['Simulations'][0]['Parameters']=[{'Path':'Applications|X|Route','Value':'Oral'}]
    elif fault=='mixed_protocol':d['Protocols'][0]['Schemas']=[{'SchemaItems':[{'ApplicationType':'Oral'}]}]
    else:d['Simulations'][0]['Compounds'].append({'Name':'Unknown'})
    with pytest.raises(ValueError):j.route_inventory(d,'target')


@pytest.fixture
def synthetic(tmp_path,monkeypatch):
    d=document()
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return {'path':name,'sha256':j.sha(p)}
    source=write('source.json',d);snapshot=write('migrated.json',d);receipt=write('receipt.json',{'snapshot':snapshot})
    record={'source':source,'snapshot':snapshot,'receipt':receipt,'compound':'target','absent_route':'Oral',
            'pairs':[{'scenario_id':'case','candidate_id':'candidate'}]}
    registry=write('registry.json',{'source_definitions':{source['sha256']:record},'clinical_outcomes_accessed':False,'native_execution_started':False})
    monkeypatch.setattr(j,'REGISTRY',registry)
    from recommendation_execution_repairs import snapshot_repair
    calls=[]
    def verify(*args,**kwargs):calls.append(kwargs);return {'snapshot':snapshot}
    monkeypatch.setattr(snapshot_repair,'_verify_receipt',verify)
    plan={'scenario_id':'case','candidate_id':'candidate','contract':{'compound':'target','route':'Oral'},
          'reasons':['Blocked:NO_SOURCE_SIMULATION_MATCHES_FIXED_EXECUTION_CONTRACT'],
          'source_migration':{'original_path':'source.json','original_sha256':source['sha256'],
              'path':'migrated.json','sha256':snapshot['sha256'],'receipt_path':'receipt.json','receipt_sha256':receipt['sha256']},
          'excluded_source_simulations':[{'simulation':s['Name'],'reason':'MULTIDRUG_SIMULATION_REQUIRES_SEPARATE_DDI_ADAPTER'} for s in d['Simulations']]}
    scenario={'scenario_id':'case','compound':'Target','normalized_design':{'route':'oral'}}
    candidate={'candidate_id':'candidate','catalog_provenance':{'source_path':'source.json','source_sha256':source['sha256']}}
    return tmp_path,d,record,plan,scenario,candidate,calls,write


def test_complete_registered_source_absence_retains_unknown(synthetic):
    root,d,record,plan,scenario,candidate,calls,_=synthetic
    refs,facts=j.prove_json_route_absence(plan,scenario,candidate,project_root=root)
    assert len(refs)==4 and len(facts)==3 and calls==[{'require_accepted':True}]
    assert all(f['scientific_applicability']=='unknown' for f in facts)


@pytest.mark.parametrize('fault',['pair','candidate','source','target','clinical_route','reason','exclusion','duplicate_exclusion','migration_hash','migration_path','receipt_hash'])
def test_registration_and_inventory_tampering_rejected(synthetic,fault):
    root,d,record,plan,scenario,candidate,_,_=synthetic
    if fault=='pair':scenario['scenario_id']='different'
    elif fault=='candidate':candidate['candidate_id']='different'
    elif fault=='source':candidate['catalog_provenance']['source_sha256']='0'*64
    elif fault=='target':plan['contract']['route']='Intravenous'
    elif fault=='clinical_route':scenario['normalized_design']['route']='iv'
    elif fault=='reason':plan['reasons']=['Blocked:EXECUTION_TIMEOUT']
    elif fault=='exclusion':plan['excluded_source_simulations'].pop()
    elif fault=='duplicate_exclusion':plan['excluded_source_simulations'][0]=dict(plan['excluded_source_simulations'][1])
    elif fault=='migration_hash':plan['source_migration']['sha256']='0'*64
    elif fault=='migration_path':plan['source_migration']['path']='source.json'
    else:plan['source_migration']['receipt_sha256']='0'*64
    with pytest.raises(ValueError):j.prove_json_route_absence(plan,scenario,candidate,project_root=root)


@pytest.mark.parametrize('both',[False,True])
def test_available_matching_route_is_never_waived_even_if_registered(synthetic,monkeypatch,both):
    root,d,record,plan,scenario,candidate,_,write=synthetic
    d['Protocols'][0]['ApplicationType']='Oral'
    new=write('migrated.json',d);record['snapshot'].update(new);plan['source_migration']['sha256']=new['sha256']
    if both:
        new=write('source.json',d);record['source'].update(new);plan['source_migration']['original_sha256']=new['sha256'];candidate['catalog_provenance']['source_sha256']=new['sha256']
    monkeypatch.setattr(j,'definitions',lambda root:{record['source']['sha256']:record})
    with pytest.raises(ValueError,match='MATCHING_ORIGINAL_ROUTE|INVENTORIES_DIFFER'):
        j.prove_json_route_absence(plan,scenario,candidate,project_root=root)


def test_all_real_56_blocked_plans_only_six_route_absences_pass():
    root=g.ROOT;base=root/'experiments/model_recommendation/execution_validation_20261001'
    jobs=json.loads((base/'engineering_repairs/repair_04/execution_plan.json').read_text())['jobs']
    chosen=[x for x in jobs if x['plan'].get('reasons')==['Blocked:NO_SOURCE_SIMULATION_MATCHES_FIXED_EXECUTION_CONTRACT']]
    assert len(chosen)==56
    accepted=[];blocked=[]
    for job in chosen:
        r=g.assess_blocked_plan(job['plan'],project_root=root)
        (accepted if r['accepted'] else blocked).append(job)
        assert r['scientific_applicability']=='unknown'
    assert len(accepted)==6 and len(blocked)==50
    assert {x['plan']['contract']['compound'] for x in accepted}=={'digoxin','voriconazole'}


def test_new_proof_not_in_candidate_implementation_hashes():
    from recommendation_execution_repairs import complete_candidate,snapshot_candidate
    names={'scripts/recommendation_execution_repairs/json_source_absence.py','scripts/recommendation_execution_repairs/engineering_gate.py'}
    assert not names.intersection(complete_candidate.IMPLEMENTATION_FILES)
    assert not names.intersection(snapshot_candidate.IMPLEMENTATION_FILES)
