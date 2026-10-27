"""Frozen mean-weight dosing keeps units, provenance and claim limits explicit."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts.recommendation_execution_repairs import clinical_contract as m
from scripts.recommendation_execution_repairs import complete_candidate, snapshot_candidate


@pytest.fixture
def weight_scene(tmp_path):
    dose={'value':4,'unit':'mg/kg','basis':'voriconazole active drug','salt':'parent voriconazole',
          'conversion_rule':m.WEIGHT_RULE,'infusion_duration_min':80}
    population={'weight_mean_kg':66.71,'weight_sd_kg':7.61,'n':24}
    protocol={'case_id':'case','compound':'voriconazole','dose':deepcopy(dose),'population':deepcopy(population)}
    source=tmp_path/'protocol.json';source.write_text(json.dumps(protocol))
    scene={'case_id':'case','scenario_id':'clinical:case','compound':'voriconazole',
        'source_protocol':{'path':'protocol.json','sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
        'execution_policy':{'allow_reference_subject_approximation':True},
        'normalized_design':{'dose':dose,'population':population,'route':'IV',
            'regimen':{'dose_times_h':[0.],'observation_origin_h':0},'sampling':{'times_h':[0,1,2]},
            'matrix':'plasma','analyte':'parent'}}
    return tmp_path,scene,source


def test_mean_weight_mass_conversion_is_explicit_nonindividual_and_nonmutating(weight_scene):
    root,scene,_=weight_scene;before=deepcopy(scene)
    result=m.scenario_contract(scene,project_root=root)
    assert result['dose_mg']==pytest.approx(266.84)
    proof=result['dose_normalization']
    assert proof['reference_weight']['statistic']=='clinical_cohort_mean'
    assert proof['reference_weight']['unit']=='kg'
    assert proof['original_scenario_sha256']==m._digest(scene)
    assert proof['source_protocol']==scene['source_protocol']
    assert proof['original_model_physiology_changed'] is False
    assert proof['strict_target_match'] is False and proof['population_simulation_performed'] is False
    assert scene==before


@pytest.mark.parametrize('change',['hash','case','compound','mean','missing_mean','rule','permission','unit','nonfinite','active_salt'])
def test_weight_conversion_requires_all_frozen_source_conditions(weight_scene,change):
    root,scene,source=weight_scene
    if change=='hash':source.write_text('{}')
    if change=='case':scene['case_id']='wrong'
    if change=='compound':scene['compound']='wrong'
    if change=='mean':scene['normalized_design']['population']['weight_mean_kg']=70
    if change=='missing_mean':scene['normalized_design']['population'].pop('weight_mean_kg')
    if change=='rule':scene['normalized_design']['dose']['conversion_rule']='Do not infer body weight.'
    if change=='permission':scene['execution_policy']['allow_reference_subject_approximation']=False
    if change=='unit':scene['normalized_design']['dose']['unit']='mg/lb'
    if change=='nonfinite':scene['normalized_design']['population']['weight_mean_kg']=float('nan')
    if change=='active_salt':scene['normalized_design']['dose']['active_moiety_dose_mg']=200
    with pytest.raises(ValueError):m.scenario_contract(scene,project_root=root)


def test_sealed_protocol_reference_is_rejected_before_read(weight_scene):
    root,scene,source=weight_scene
    sealed=root/'sealed';sealed.mkdir();source.rename(sealed/'protocol.json')
    scene['source_protocol']['path']='sealed/protocol.json'
    with pytest.raises(ValueError,match='sealed'):m.scenario_contract(scene,project_root=root)


def test_repeated_equal_weight_based_doses_are_normalized_but_conflicts_rejected(weight_scene):
    root,scene,_=weight_scene
    scene['normalized_design']['regimen']['administrations']=[
        {'time_h':0,'dose':4,'dose_unit':'mg/kg'},{'time_h':12,'dose':4,'dose_unit':'mg/kg'}]
    result=m.scenario_contract(scene,project_root=root)
    assert result['administration_times_min']==[0,720]
    scene['normalized_design']['regimen']['administrations'][1]['dose']=6
    with pytest.raises(ValueError,match='CONFLICTING_WEIGHT'):m.scenario_contract(scene,project_root=root)


def test_existing_frozen_salt_conversion_remains_exact_and_unsupported_mapping_is_detailed():
    scene={'case_id':'salt','compound':'metformin','normalized_design':{
        'dose':{'value':250,'unit':'mg','active_moiety_dose_mg':195},'route':'oral',
        'regimen':{'dose_times_h':[0.],'observation_origin_h':0},'sampling':{'times_h':[0,1,2]},
        'matrix':'plasma','analyte':'parent'},'protocol_fields':{'protocol':{'salt_conversion':{
            'factor':.78,'source_file':'bound-source.json','source_file_sha256':'a'*64}}}}
    assert m.scenario_contract(scene)['dose_mg']==195
    scene['normalized_design']['dose']={'value':150,'unit':'mg','conversion_rule':'Freeze salt-to-native mass mapping.'}
    with pytest.raises(ValueError,match='parent molecular weight alone'):m.scenario_contract(scene)


def test_both_adapters_share_and_pin_the_identical_normalizer():
    assert complete_candidate.scenario_contract is snapshot_candidate.scenario_contract
    path='scripts/recommendation_execution_repairs/clinical_contract.py'
    assert path in complete_candidate.IMPLEMENTATION_FILES and path in snapshot_candidate.IMPLEMENTATION_FILES
