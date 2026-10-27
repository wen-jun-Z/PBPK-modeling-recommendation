"""Outcome-free reference selection is consistent across preserved formats."""
from copy import deepcopy

import pytest

from scripts.recommendation_execution_repairs import snapshot_candidate as m
from scripts.recommendation_execution_repairs import complete_candidate as complete


@pytest.fixture
def designs(monkeypatch):
    snapshot={'Formulations':[{'Name':'Solution'},{'Name':'XR tablet'}],
              'Simulations':[
                  {'Name':'solution fed source','Compounds':[{'Name':'Quetiapine',
                      'Protocol':{'Name':'P1','Formulations':[{'Name':'Solution','Key':'Formulation'}]}}]},
                  {'Name':'XR source','Compounds':[{'Name':'Quetiapine',
                      'Protocol':{'Name':'P2','Formulations':[{'Name':'XR tablet','Key':'Formulation'}]}}]}]}
    contract={'compound':'quetiapine','route':'Oral','dose_mg':50.,
              'formulation':'Two25mg Seroquel film-coated reference tablets',
              'food':'Not explicitly stated; do not infer fasting.',
              'administration_times_min':[0.], 'native_binding':{}}
    seen=[]
    def legacy(s,c,*,source_sha256):
        seen.append(deepcopy(c));options=[];rejected=[]
        for sim,dose in zip(s['Simulations'],[25.,50.]):
            form=m.original._form(sim['Compounds'][0]['Protocol']['Formulations'][0]['Name'])
            if m.original._form(c.get('formulation')) not in (None,form):
                rejected.append({'simulation':sim['Name'],'reason':'SOURCE_FORMULATION_CLASS_DIFFERS'})
                continue
            options.append({'name':sim['Name'],'compound_name':'Quetiapine','source_dose_mg':dose,
                            'source_formulation_class':form,'output_path':'parent plasma'})
        return options,rejected
    monkeypatch.setattr(m,'_source_options',legacy)
    return snapshot,contract,seen


def native_metadata(name,form,dose):
    prefix='Applications|P|F|Application_1'
    return {'name':name,'simulation_id':name,'saved_type':'IndividualSimulation',
        'species':['Human'],'model_configuration':'4Comp|Human','compounds':['Quetiapine'],
        'formulations':[{'name':form}],
        'event_parameters':[{'path':prefix+'|ProtocolSchemaItem|Dose','id':'dose','value':dose/1e6,'dimension':'Mass'},
                            {'path':prefix+'|ProtocolSchemaItem|Start time','id':'time','value':0.,'dimension':'Time'}],
        'administrations':[{'event_prefix':prefix,'application_type':'Oral',
                           'administered_compounds':['Quetiapine'],'assignment_targets':['exact parent path'],
                           'compound_association_complete':True}]}


def test_reference_release_preference_agrees_between_snapshot_and_native(designs):
    snapshot,contract,seen=designs
    before=deepcopy((snapshot,contract))
    options,_=m.reference_source_options(snapshot,contract,source_sha256='source',reference_allowed=True)
    native={'simulations':[native_metadata('solution fed source','Solution',25.),
                           native_metadata('XR source','XR tablet',50.)]}
    expected,_=complete.select_source_simulation(native,contract)
    assert options[0]['name']==expected[0]['metadata']['name']=='solution fed source'
    design=options[0]['source_selection_design']
    assert design['target_release']=='immediate_release' and design['source_release']=='solution'
    assert design['target_food_state']=='unknown' and seen[0]['food']=='unknown'
    assert design['strict_target_match'] is False
    assert design['source_formulation_changed_by_selection'] is False
    assert design['approximations'] and before==(snapshot,contract)


def test_reference_permission_is_required_to_relax_only_formulation_class(designs):
    snapshot,contract,seen=designs
    options,rejected=m.reference_source_options(snapshot,contract,source_sha256='source',reference_allowed=False)
    assert [x['name'] for x in options]==['XR source']
    assert rejected[0]['reason']=='SOURCE_FORMULATION_CLASS_DIFFERS'
    assert seen[0]['formulation']==contract['formulation']
    assert options[0]['source_selection_design']['reference_subject_approximation_explicitly_permitted'] is False


def test_modified_release_prefers_existing_xr_without_mutating_source(designs):
    snapshot,contract,_=designs;contract['formulation']='Seroquel XR extended-release tablet'
    options,_=m.reference_source_options(snapshot,contract,source_sha256='source',reference_allowed=True)
    assert options[0]['name']=='XR source'
    assert options[0]['source_selection_design']['target_release']=='modified_release'
    assert options[0]['source_selection_design']['source_release']=='modified_release'
    assert options[0]['source_selection_design']['original_source_formulations']==[{'Name':'XR tablet'}]


def test_unspecified_tablet_release_remains_unknown_and_reported(designs):
    snapshot,contract,_=designs;contract['formulation']='50 mg tablet; brand not known'
    options,_=m.reference_source_options(snapshot,contract,source_sha256='source',reference_allowed=True)
    design=options[0]['source_selection_design']
    assert design['target_release']=='unknown'
    assert 'source_and_target_release_behavior_differs_or_is_unknown; reference_only' in design['approximations']


def test_release_helpers_are_pinned_as_execution_dependencies():
    assert 'scripts/recommendation_execution_repairs/complete_candidate.py' in m.IMPLEMENTATION_FILES


def test_parent_alias_requires_exact_source_and_chemical_mass_proof():
    from recommendation_execution_repairs.snapshot_candidate import source_parent_identity
    from recommendation_candidate_executor import Blocked
    digest='0635dcc6fd3f9f45c91b0f5ca480159926f0f57772c9af4e37ed78def0e5ea02'
    contract={'compound':'bupropion','case_id':'bupropion_teaima2020_wellbutrin_sr_150mg',
        'salt_conversion':{'parent_model_source':{'sha256':digest},'parent_molecular_weight_g_per_mol':'239.741',
                           'parent_definition_evidence':[{'compound_name':'Bupropion_FM'}]}}
    assert source_parent_identity(contract,digest)['native_name']=='Bupropion_FM'
    assert source_parent_identity(contract,'other_source') is None
    contract['salt_conversion']['parent_molecular_weight_g_per_mol']='250'
    with pytest.raises(Blocked,match='CHEMISTRY_PROOF'):source_parent_identity(contract,digest)
