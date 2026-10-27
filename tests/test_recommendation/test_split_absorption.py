"""Source-bound split absorption cannot be suppressed, doubled, or retimed."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from recommendation_execution_repairs import split_absorption as s,complete_candidate as c
BASE=ROOT/'experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_native_bridge_followup_20261002'

@pytest.fixture(scope='module')
def case():
    row=json.loads((BASE/'split_absorption_actual_source_checks.json').read_text())['rows'][0]
    prior=json.loads((ROOT/'experiments/model_recommendation/execution_validation_20261001/native'/row['job_id']/'execution_plan.json').read_text())
    metadata=row['metadata'];inventory={'parameters':[]}
    for p in metadata['event_parameters']:
        if not c.native_pkml.ADMINISTRATION.fullmatch(p['path']):continue
        inventory['parameters'].append({'id':p['id'],'path':p['path'],'value':float(p['value']),
                                       'unit':'kg' if p['dimension']=='Mass' else 'min'})
    return row,prior['contract'],inventory


def test_actual_source_contract_and_native_dose_mass_conservation(case):
    row,contract,inventory=case
    patches,events=s.build_split_patches(inventory,contract,row['metadata'],row['policy'],project_root=ROOT)
    assert len(patches)==4 and len(events)==2
    assert [r['dose_mg'] for r in events]==[25.36,14.64]
    assert [r['start_time_min'] for r in events]==[0,138]
    assert sum(r['dose_mg'] for r in events)==40
    assert all(r['target_administration_index']==0 for r in events)


@pytest.mark.parametrize('kind',['fraction','lag','dose','event_id','metadata','clinical_dose','clinical_timing','source_identity'])
def test_source_or_contract_changes_are_rejected(case,kind):
    row,contract,inventory=deepcopy(case);policy=row['policy'];metadata=row['metadata']
    if kind=='fraction':policy['events'][0]['fraction']=.5
    elif kind=='lag':policy['events'][1]['target_start_time_min']=0
    elif kind=='dose':policy['events'][1]['target_dose_mg']=40
    elif kind=='event_id':policy['events'][0]['dose_parameter_id']='invented'
    elif kind=='metadata':metadata['event_parameters'][0]['value']='999'
    elif kind=='clinical_dose':contract['dose_mg']=80
    elif kind=='clinical_timing':contract['administration_times_min']=[0,138]
    else:policy['source_original']['sha256']='0'*64
    with pytest.raises(ValueError):s.revalidate_contract(policy,contract,metadata,project_root=ROOT)


@pytest.mark.parametrize('kind',['missing_input','wrong_id','wrong_mass','wrong_time','duplicate'])
def test_actual_native_inventory_changes_rejected(case,kind):
    row,contract,inventory=deepcopy(case);p=inventory['parameters']
    if kind=='missing_input':p.pop()
    elif kind=='wrong_id':p[0]['id']='replacement'
    elif kind=='wrong_mass':next(r for r in p if r['unit']=='kg')['value']*=2
    elif kind=='wrong_time':next(r for r in p if r['unit']=='min')['value']+=1
    else:p.append(deepcopy(p[0]))
    with pytest.raises((ValueError,KeyError)):s.build_split_patches(inventory,contract,row['metadata'],row['policy'],project_root=ROOT)


@pytest.fixture
def readback(case):
    row,contract,inventory=case
    patches,_=s.build_split_patches(inventory,contract,row['metadata'],row['policy'],project_root=ROOT)
    by_path={r['path']:r for r in inventory['parameters']}
    return {'parameter_changes':[{'path':p['path'],'id':by_path[p['path']]['id'],
        'after':{'value':p['value'],'unit':p['unit']}} for p in patches]},row['policy']


def test_two_internal_inputs_read_back_as_one_clinical_dose(readback):
    native,policy=readback;a=s.verify_native_readback(native,policy)
    assert a['accepted'] and a['clinical_administration_count']==1 and a['internal_absorption_event_count']==2


@pytest.mark.parametrize('kind',['zero_second','double_dose','zero_lag','missing','duplicate','id','nan'])
def test_native_wrong_absorption_is_not_a_valid_prediction(readback,kind):
    native,policy=deepcopy(readback);rows=native['parameter_changes'];last_dose=[r for r in rows if r['path'].endswith('|Dose')][-1]
    if kind=='zero_second':last_dose['after']['value']=0
    elif kind=='double_dose':last_dose['after']['value']=40
    elif kind=='zero_lag':[r for r in rows if r['path'].endswith('|Start time')][-1]['after']['value']=0
    elif kind=='missing':rows.pop()
    elif kind=='duplicate':rows.append(deepcopy(rows[0]))
    elif kind=='id':rows[0]['id']='other'
    else:rows[0]['after']['value']=float('nan')
    with pytest.raises(ValueError):s.verify_native_readback(native,policy)
