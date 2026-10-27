"""An exact source/request gap never upgrades all source uses or fabricates labels."""
from copy import deepcopy
import importlib.util,json,sys
from pathlib import Path
import pytest

STAGE=Path(__file__).resolve().parents[2]
ROOT=next(p for p in (STAGE,*STAGE.parents) if (p/'src/pbpk_kb').is_dir())
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
spec=importlib.util.spec_from_file_location('recommendation_execution_repairs.source_contract_gap',
    STAGE/'scripts/recommendation_execution_repairs/source_contract_gap.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.fixture(params=[0,1,2])
def case(monkeypatch,request):
    registry=m._registry(ROOT);entry=deepcopy(registry['cases'][request.param])
    scenario={'original':'frozen scenario'};candidate={'original':'frozen candidate'}
    plan={k:deepcopy(entry[k]) for k in ('scenario_id','candidate_id','scenario_sha256','candidate_sha256')}
    plan.update(status='blocked',reasons=entry['original_blocked_reasons'],contract=deepcopy(entry['expected_contract']),
                sealed_observations_accessed=False,fitting_performed=False)
    monkeypatch.setattr(m,'_registry',lambda root:registry)
    calls=[]
    def evidence(e,reg,root):
        calls.append(e);return scenario,candidate,{'matching_capsule_fed_source_simulations':[]},[m.REGISTRY]
    monkeypatch.setattr(m,'_evidence',evidence)
    return entry,plan,scenario,candidate,calls


@pytest.mark.parametrize('route',[None,'blocked_json_native_bridge'])
def test_exact_blocked_pair_stays_unknown_in_full_denominator(case,route):
    _,plan,scenario,candidate,calls=case
    if route is not None:plan['adapter_route']=route
    result=m.assess_plan_gap(plan,scenario,candidate,project_root=ROOT)
    assert result['accepted'] and len(calls)==1
    assert result['scientific_applicability']=='unknown' and result['inclusion_in_full_candidate_denominator']
    assert result['original_source_retained_for_reference']
    assert not any(result[k] for k in ('source_execution_verified','numerical_prediction_available',
        'matched_prediction_eligible','direct_reuse_eligible','scientific_failure_inferred'))


@pytest.mark.parametrize('fault',['other_case','other_candidate','pair_hash','scenario_hash','dose','formulation',
    'food','regimen','status','reason','outcomes','fitting','caller_scenario','caller_candidate'])
def test_nearby_uses_or_mutated_contract_never_inherit_the_exception(case,fault):
    _,plan,scenario,candidate,_=case
    if fault=='other_case':plan['scenario_id']='clinical:other'
    if fault=='other_candidate':plan['candidate_id']='other'
    if fault=='pair_hash':plan['candidate_sha256']='a'*64
    if fault=='scenario_hash':plan['scenario_sha256']='a'*64
    if fault=='dose':plan['contract']['dose_mg']=100
    if fault=='formulation':plan['contract']['formulation']='Solution'
    if fault=='food':plan['contract']['food']='fasted'
    if fault=='regimen':plan['contract']['administration_times_min'].pop()
    if fault=='status':plan['status']='ready'
    if fault=='reason':plan['reasons']=['Blocked:other engineering failure']
    if fault=='outcomes':plan['sealed_observations_accessed']=True
    if fault=='fitting':plan['fitting_performed']=True
    if fault=='caller_scenario':scenario={'changed':True}
    if fault=='caller_candidate':candidate={'changed':True}
    result=m.assess_plan_gap(plan,scenario,candidate,project_root=ROOT)
    assert not result['accepted'] and result['category']=='engineering_repair_required'


def source_document(entry):
    document={'Version':77,'Simulations':[{'Name':entry['source_simulation'],
        'Compounds':[deepcopy(entry['source_compound_binding'])]},{'Name':'control','Compounds':[{'Name':'Felodipine'}]}],
        'Protocols':[deepcopy(entry['source_protocol'])],'Formulations':deepcopy(entry['source_formulations']),
        'Compounds':[{'Name':'Itraconazole','Solubility':[{'Name':'Solution fasted (Taupitz et al. 2013)',
            'Parameters':[{'Name':'Solubility at reference pH','Value':8.,'Unit':'mg/l'}]}]}]}
    if entry['candidate_id']=='3df9ee3a-afb1-5c3b-a652-dfa3ff9c7390':
        document['Simulations'][1]['Name']='Prueksaritanont 2017 - DDI Control'
        document['Simulations'].reverse()
    return document


def test_complete_source_inventory_has_no_matched_configuration(case):
    entry,*_=case;facts=m._original_facts(source_document(entry),entry)
    assert len(facts['all_original_simulations'])==2
    assert facts['all_parent_containing_simulations']==[entry['source_simulation']]
    assert facts['matching_capsule_fed_source_simulations']==[]
    assert facts['source_clinical_formulation_inferred'] is False


@pytest.mark.parametrize('fault',['matching_simulation_added','target_simulation_removed','selection','protocol',
    'formulation_type','solubility','unit','missing_parent','duplicate_simulation','snapshot_version'])
def test_all_source_simulations_and_exact_selected_parameters_are_checked(case,fault):
    entry,*_=case;doc=source_document(entry)
    target=next(s for s in doc['Simulations'] if s['Name']==entry['source_simulation'])
    if fault=='matching_simulation_added':
        addition=deepcopy(target);addition['Name']='Capsule fed';doc['Simulations'].append(addition)
    if fault=='target_simulation_removed':doc['Simulations']=[s for s in doc['Simulations'] if s is not target]
    if fault=='selection':next(a for a in target['Compounds'][0]['Alternatives'] if a['GroupName']=='COMPOUND_SOLUBILITY')['AlternativeName']='Capsule fed'
    if fault=='protocol':doc['Protocols'][0]['Schemas'][0]['SchemaItems'][0]['Parameters'][1]['Value']=100
    if fault=='formulation_type':doc['Formulations'][0]['FormulationType']='Formulation_Tablet_Weibull'
    if fault=='solubility':doc['Compounds'][0]['Solubility'][0]['Parameters'][0]['Value']=0.7
    if fault=='unit':doc['Compounds'][0]['Solubility'][0]['Parameters'][0]['Unit']='g/l'
    if fault=='missing_parent':doc['Compounds']=[]
    if fault=='duplicate_simulation':doc['Simulations'].append(deepcopy(doc['Simulations'][0]))
    if fault=='snapshot_version':doc['Version']=81
    with pytest.raises(ValueError):m._original_facts(doc,entry)


def test_registry_file_tampering_is_rejected_before_context(tmp_path,monkeypatch):
    p=tmp_path/'registry.json';p.write_text('{}')
    monkeypatch.setattr(m,'REGISTRY',{'path':'registry.json','sha256':'a'*64})
    with pytest.raises(ValueError):m._registry(tmp_path)


@pytest.mark.parametrize('fault',['different_project','different_native_sha','different_simulation','different_blob','different_name','ambiguous_project'])
def test_same_name_elsewhere_cannot_replace_the_original_native_source(case,monkeypatch,fault):
    from types import SimpleNamespace
    import recommendation_execution_repairs as package
    entry,*_=case;n=entry['native_source']
    native={k:n[k] for k in ('path','sha256')}
    sim={'id':n['simulation_id'],'name':entry['source_simulation'],'source_blob_sha256':n['source_blob_sha256']}
    if fault=='different_project':native['path']='another/same_name.pksim5'
    if fault=='different_native_sha':native['sha256']='a'*64
    if fault=='different_simulation':sim['id']='foreign'
    if fault=='different_blob':sim['source_blob_sha256']='a'*64
    if fault=='different_name':sim['name']='foreign'
    record={'native_source':native,'allowed_simulations':[sim]}
    candidates=[record,deepcopy(record)] if fault=='ambiguous_project' else [record]
    def resolve(provenance,**kwargs):
        assert provenance=={'source_path':entry['source_original']['path'],
                            'source_sha256':entry['source_original']['sha256']}
        return {'candidates':candidates}
    monkeypatch.setattr(package,'json_native_source_proof',SimpleNamespace(resolve_native_sources=resolve),raising=False)
    with pytest.raises(ValueError):m._native_facts(entry,ROOT)


@pytest.fixture
def third_native(monkeypatch):
    import xml.etree.ElementTree as ET
    from types import SimpleNamespace
    import recommendation_execution_repairs as package
    from recommendation_execution_repairs import source_pkml,mixed_application_expansion as mixed
    entry=deepcopy(m._registry(ROOT)['cases'][2]);native=entry['native_source']
    selected={'id':native['simulation_id'],'name':entry['source_simulation'],
              'source_blob_sha256':native['source_blob_sha256']}
    proof={'native_source':{k:native[k] for k in ('path','sha256')},'allowed_simulations':[selected],
           'proof_refs':[m.REGISTRY]}
    monkeypatch.setattr(package,'json_native_source_proof',SimpleNamespace(
        resolve_native_sources=lambda *a,**k:{'candidates':[proof]}),raising=False)
    model=ET.Element('Model');ET.SubElement(model,'Root',name=entry['source_simulation'])
    paths={};ids={};hashes={};nodes=[]
    for saved in entry['expected_native_release_events']:
        path=saved['path'];node=ET.Element('EventGroup',name=path.rsplit('|',1)[-1]);children=ET.SubElement(node,'Children')
        schema=ET.SubElement(children,'Container',name='ProtocolSchemaItem')
        ET.SubElement(children,'MoleculeAmount',name='Itraconazole')
        for attrs in saved['start_events']:ET.SubElement(children,'Event',**attrs)
        for attrs in saved['assignments']:
            ET.SubElement(children,'EventAssignment',**attrs)
            destination=ET.Element('Parameter',id=attrs['changedEntity'])
            paths[attrs['objectPath']]=destination;ids[attrs['changedEntity']]=destination
        for name,key in [('Dose','native_dose_kg'),('Start time','native_start_time_min')]:
            paths[path+'|ProtocolSchemaItem|'+name]=ET.SubElement(schema,'Parameter',name=name,value=str(saved[key]))
        paths[path]=node;hashes[id(node)]=saved['source_node_sha256'];nodes.append(node)
    monkeypatch.setattr(mixed.base,'_original_document',lambda *a:model)
    monkeypatch.setattr(source_pkml,'wrap_preserved_model',lambda *a:(model,None))
    monkeypatch.setattr(mixed.base,'_models',lambda *a:model)
    monkeypatch.setattr(mixed.base,'_inventory',lambda *a:(paths,ids,{}))
    monkeypatch.setattr(mixed.base,'_formula_inventory',lambda *a:({},{}))
    monkeypatch.setattr(mixed.base,'_scalar',lambda node,*a:float(node.get('value')))
    monkeypatch.setattr(source_pkml,'element_sha',lambda node:hashes[id(node)])
    return entry,model,paths,ids,nodes,hashes


def test_exact_registered_five_solution_events_replay(third_native):
    entry,*_=third_native
    events,refs=m._native_facts(entry,ROOT)
    assert len(events)==5 and refs==[m.REGISTRY]


@pytest.mark.parametrize('fault',['four_events','six_events','release_transport','stop_event','repeated_event',
    'foreign_destination','wrong_destination_id','missing_assignment','extra_assignment','wrong_dose','wrong_time',
    'tablet_state','changed_node','changed_registered_event','wrong_source_count'])
def test_third_native_configuration_cannot_be_relabeled_or_broadened(third_native,fault):
    import xml.etree.ElementTree as ET
    entry,model,paths,ids,nodes,hashes=third_native
    node=nodes[-1];path=entry['expected_native_release_events'][-1]['path']
    children=node.find('Children')
    if fault=='four_events':del paths[path]
    elif fault=='six_events':paths[path+'_extra']=deepcopy(node)
    elif fault=='release_transport':ET.SubElement(children,'Transport')
    elif fault=='stop_event':ET.SubElement(children,'Event',name='Application_StopEvent')
    elif fault=='repeated_event':node.find('.//Event').set('oneTime','0')
    elif fault=='foreign_destination':node.find('.//EventAssignment').set('objectPath','foreign|drug')
    elif fault=='wrong_destination_id':node.find('.//EventAssignment').set('changedEntity','foreign')
    elif fault=='missing_assignment':children.remove(node.find('.//EventAssignment'))
    elif fault=='extra_assignment':ET.SubElement(children,'EventAssignment')
    elif fault=='wrong_dose':paths[path+'|ProtocolSchemaItem|Dose'].set('value','0.0001')
    elif fault=='wrong_time':paths[path+'|ProtocolSchemaItem|Start time'].set('value','6000')
    elif fault=='tablet_state':ET.SubElement(children,'Parameter',name='TabletIsActive')
    elif fault=='changed_node':hashes[id(node)]='a'*64
    elif fault=='changed_registered_event':entry['expected_native_release_events'][-1]['native_start_time_min']=6000.
    elif fault=='wrong_source_count':entry['expected_native_release_events'].pop()
    with pytest.raises(ValueError):m._native_facts(entry,ROOT)
