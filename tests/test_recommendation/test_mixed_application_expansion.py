"""Complete target app clones cannot alter original mixed-drug Model/Settings."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.recommendation_execution_repairs import mixed_application_expansion as mixed
from scripts.recommendation_execution_repairs import repeated_application as base
from scripts.recommendation_execution_repairs import source_pkml

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('single_app_fixture',Path(__file__).with_name('test_repeated_application.py'))
fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)


def document():
    original=fixture_module.document()
    doc,_=base.expand_document(original,compound_name='Pioglitazone',event_prefix=fixture_module.PREFIX,
                                administration_times_min=[0.,1440.],dose_mg=30.)
    root=doc.find('.//Model/Root/Children')
    other=ET.SubElement(root,'Container',id='otherdrug',name='OtherDrug')
    children=ET.SubElement(other,'Children')
    ET.SubElement(children,'Parameter',id='other_initial',name='Initial mass',dim='Amount',value='0')
    ET.SubElement(children,'Parameter',id='other_total',name='Total drug mass',dim='Amount',value='0.01')
    app=ET.SubElement(root,'EventGroup',id='otherapp',name='OtherApplication',containerType='Application',eventGroupType='Intravenous')
    ch=ET.SubElement(app,'Children');schema=ET.SubElement(ch,'Container',id='otherschema',name='ProtocolSchemaItem')
    sc=ET.SubElement(schema,'Children')
    ET.SubElement(sc,'Parameter',id='other_dose',name='Dose',dim='Mass',value='0.001')
    ET.SubElement(sc,'Parameter',id='other_start',name='Start time',dim='Time',value='100')
    ET.SubElement(ch,'MoleculeAmount',id='other_amount',name='OtherDrug',dim='Amount',value='0')
    return doc


def expand():
    source=document();contract={'route':'Oral','dose_mg':30.,'administration_times_min':[0.,720.,1440.,2160.]}
    result,plan=mixed.expand_document(source,target='Pioglitazone',contract=contract)
    return source,result,plan


def test_all_original_mixed_applications_and_settings_survive_exactly():
    source,result,plan=expand();proof=mixed.audit_expansion(source,result,plan)
    assert proof['accepted'],proof
    assert plan['source_target_count']==2 and plan['source_total_application_count']==3
    assert plan['target_count']==4 and len(plan['added_applications'])==2
    original_model=base._models(source);new_model=base._models(result)
    _,original_ids,_=base._inventory(original_model);_,new_ids,_=base._inventory(new_model)
    for name in ('otherapp','other_initial','other_total','other_dose','other_start'):
        assert base._canonical(original_ids[name])==base._canonical(new_ids[name])
    for old in [n for n in original_model.iter('EventGroup') if n.get('containerType')=='Application']:
        assert base._canonical(old)==base._canonical(new_ids[old.get('id')])
    assert base._canonical(source)==base._canonical(document())


@pytest.mark.parametrize('fault',['other_dose','other_total','other_initial','original_target_dose','settings',
    'clone_transport','clone_internal','clone_mass','sum_other_drug','sum_alias','new_id','string_map','extra_application'])
def test_full_inverse_audit_rejects_every_unregistered_change(fault):
    source,result,plan=expand();model=base._models(result);paths,ids,_=base._inventory(model)
    if fault in {'other_dose','other_total','other_initial'}:ids[fault].set('value','99')
    if fault=='original_target_dose':ids['dose'].set('value','0.9')
    if fault=='settings':result.find('.//Solver').set('relativeTolerance','0.1')
    branch=paths[model.find('Root').get('name')+'|'+plan['added_applications'][-1]['prefix']]
    if fault=='clone_transport':next(branch.iter('Transport')).set('targetAmount','other_amount')
    if fault=='clone_internal':next(n for n in branch.iter('Parameter') if n.get('name')=='Dissolution rate').set('value','0.43')
    if fault=='clone_mass':next(n for n in branch.iter('Parameter') if n.get('name')=='Dose').set('value','0.004')
    formula=next(n for n in model.find('FormulaCache/Formulas') if n.get('id')==plan['target_mass_formula_id'])
    if fault=='sum_other_drug':formula.find('Paths')[-1].set('path','Original|OtherDrug|Total drug mass')
    if fault=='sum_alias':formula.find('Paths')[-1].set('as','P_1')
    if fault=='new_id':branch.set('id','otherapp')
    if fault=='string_map':model.find('FormulaCache/StringMap')[0].set('s','corrupted')
    if fault=='extra_application':model.find('Root/Children').append(deepcopy(ids['otherapp']))
    assert not mixed.audit_expansion(source,result,plan)['accepted']


def test_template_is_last_original_target_and_generated_ids_are_deterministic():
    source,result,plan=expand()
    second,second_plan=mixed.expand_document(source,target='Pioglitazone',contract=plan['contract'])
    assert base._canonical(result)==base._canonical(second) and plan==second_plan
    assert plan['template_prefix'].endswith('Application_2')
    assert [r['name'] for r in plan['added_applications']]==['Application_3','Application_4']
    assert plan['added_applications'][0]['start_min']==1440.


def test_same_application_basename_in_other_drug_formula_is_not_a_target_dependency():
    doc=document();model=base._models(doc);_,ids,_=base._inventory(model)
    ids['otherapp'].set('name','Application_2')
    schema=ids['otherschema'].find('Children')
    ET.SubElement(schema,'Parameter',id='other_mass',name='DrugMass',dim='Amount',value='0.01')
    ids['other_total'].attrib.pop('value');ids['other_total'].set('formula','other_sum')
    formula=ET.SubElement(model.find('FormulaCache/Formulas'),'Formula',id='other_sum',formula='P_1',dim='Amount')
    group=ET.SubElement(formula,'Paths')
    ET.SubElement(group,'Path',path='Original|Application_2|ProtocolSchemaItem|DrugMass',**{'as':'P_1'})
    expanded,plan=mixed.expand_document(doc,target='Pioglitazone',
        contract={'route':'Oral','dose_mg':30.,'administration_times_min':[0.,720.,1440.]})
    assert mixed.audit_expansion(doc,expanded,plan)['accepted']
    formula.find('Paths')[0].set('path','Original|'+fixture_module.PREFIX.replace('Application_1','Application_2')+'|ProtocolSchemaItem|DrugMass')
    with pytest.raises(ValueError,match='EXTERNAL_TEMPLATE_DEPENDENCY'):
        mixed.expand_document(doc,target='Pioglitazone',
            contract={'route':'Oral','dose_mg':30.,'administration_times_min':[0.,720.,1440.]})


@pytest.mark.parametrize('count',[1,2,31])
def test_only_missing_bounded_target_events_may_be_added(count):
    with pytest.raises(ValueError):
        mixed.expand_document(document(),target='Pioglitazone',
            contract={'route':'Oral','dose_mg':30.,'administration_times_min':[720.*i for i in range(count)]})


def test_unknown_topology_or_nonexact_trigger_cannot_be_repeated():
    doc=document();next(n for n in doc.iter('Formula') if n.get('id')=='begin').set('formula','Time>=StartTime')
    with pytest.raises(ValueError,match='STARTTIME_TRIGGER'):
        mixed.expand_document(doc,target='Pioglitazone',contract={'route':'Oral','dose_mg':30.,'administration_times_min':[0.,720.,1440.]})


def test_actual_clarithromycin_nine_target_apps_expand_to_ten_without_native_execution():
    plan=json.loads((ROOT/'runtime/deployment/complete_candidate_preflight_20261001_after_repairs/attempt_02/plans/011.json').read_text())
    ref={'path':plan['source_path'],'sha256':plan['source_sha256'],
         'simulation_id':'VvgW5sQZBUyZ_8rvPpQkIQ',
         'source_blob_sha256':'7f8092b31662d7e73aa1e6dbc53351ba9488d80f185886b4eb3ebeeed596d9da'}
    saved=base._original_document(ref,ROOT);metadata=source_pkml.describe_saved_simulation(saved)
    policy=mixed.prepare_source_expansion(ref['path'],ref['sha256'],ref['simulation_id'],ref['source_blob_sha256'],
            metadata,'Clarithromycin',plan['contract'])
    assert policy['source_target_event_count']==9 and policy['executable_target_event_count']==10
    assert policy['source_total_event_count']==10 and policy['executable_total_event_count']==11
    doc,_=source_pkml.wrap_preserved_model(saved)
    expanded,edit=mixed.expand_document(doc,target='Clarithromycin',contract=plan['contract'])
    proof=mixed.audit_expansion(doc,expanded,edit)
    assert proof['accepted'],proof
    assert proof['all_original_applications_and_other_drugs_unchanged']
