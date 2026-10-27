"""Old resource limits cannot authorize changed source/expansion identities."""
from copy import deepcopy
import hashlib,importlib.util,json,sys
from pathlib import Path
import pytest

STAGED=Path(__file__).resolve().parents[2]
PROJECT=next(p for p in (STAGED,*STAGED.parents) if (p/'src/pbpk_kb').is_dir())
sys.path[:0]=[str(PROJECT/'scripts'),str(PROJECT/'src')]
from score_recommendation_execution import Files
from recommendation_execution_repairs import complete_candidate
spec=importlib.util.spec_from_file_location('recommendation_execution_repairs.historical_mixed_scoring',
    STAGED/'scripts/recommendation_execution_repairs/historical_mixed_scoring.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    f=Files(tmp_path)
    def write(path,value):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value))
        return {'path':path,'sha256':f.sha(p)}
    mixed='scripts/recommendation_execution_repairs/mixed_application_expansion.py'
    native='scripts/recommendation_execution_repairs/native_pkml.py'
    adapter='scripts/recommendation_execution_repairs/complete_candidate.py'
    code=[write(p,{'producer':p}) for p in (mixed,native,adapter)]
    frozen={r['path']:r['sha256'] for r in code}
    old={**frozen,native:'a'*64}
    source=write('source.pksim5',{'source':True})
    policy={'version':m.VERSION,'target':'Drug','contract':{'dose':5}}
    plan={'adapter_route':'complete_original_pksim5','implementation_sha256':deepcopy(old),
          'application_expansion':policy,'source_path':source['path'],'source_sha256':source['sha256'],
          'source_simulation_id':'original-simulation','source_model_blob_sha256':'b'*64}
    receipt={'implementation_sha256':{p:old[p] for p in (mixed,native)},
        'source_reference':{**source,'simulation_id':plan['source_simulation_id'],
                            'source_blob_sha256':plan['source_model_blob_sha256']},
        'source_pkml':write('job/export.pkml',{'original':True}),
        'pkml':write('job/expanded.pkml',{'expanded':True}),
        'files':{'job/policy.json':write('job/policy.json',policy)['sha256']},
        **{k:False for k in ('ode_executed','fitting_performed','sealed_observations_accessed',
                            'original_applications_modified','scientific_applicability_claimed')}}
    protocol_path='experiments/model_recommendation/execution_validation_20261001/protocol.json'
    write(protocol_path,{'code':code})
    monkeypatch.setattr(m.mixed,'IMPLEMENTATION_FILES',(mixed,native))
    monkeypatch.setattr(complete_candidate,'IMPLEMENTATION_FILES',(mixed,native,adapter))
    calls=[]
    def binding(record,actual_frozen,**kwargs):
        calls.append((deepcopy(record),actual_frozen,kwargs))
        assert record==plan and actual_frozen==frozen
        return old,{mixed,native,adapter}
    monkeypatch.setattr(m,'binding_profile',binding)
    return locals()


def test_registered_native_wrapper_drift_still_authenticates_entire_original_plan(fixture):
    p=fixture;m._authenticate(p['f'],p['plan'],p['receipt'],p['policy'])
    assert len(p['calls'])==1
    call=p['calls'][0]
    assert call[2]['adapter_path']==p['adapter']
    assert set(call[2]['current_required'])==set(p['old'])
    assert p['receipt']['implementation_sha256'][p['native']]=='a'*64


@pytest.mark.parametrize('fault', ['missing_producer','extra_producer','receipt_sha','other_drift',
    'source_path','source_sha','simulation','blob','policy','flags','source_file','evidence_file',
    'code_file','duplicate_code','unsupported_route'])
def test_unregistered_drift_or_changed_evidence_is_rejected(fixture,fault):
    p=fixture;r=p['receipt'];plan=p['plan'];policy=p['policy']
    if fault=='missing_producer':r['implementation_sha256'].pop(p['native'])
    if fault=='extra_producer':r['implementation_sha256']['extra.py']='c'*64
    if fault=='receipt_sha':r['implementation_sha256'][p['native']]='d'*64
    if fault=='other_drift':
        p['old'][p['mixed']]=plan['implementation_sha256'][p['mixed']]=r['implementation_sha256'][p['mixed']]='e'*64
    if fault=='source_path':r['source_reference']['path']='different.pksim5'
    if fault=='source_sha':r['source_reference']['sha256']='e'*64
    if fault=='simulation':r['source_reference']['simulation_id']='foreign'
    if fault=='blob':r['source_reference']['source_blob_sha256']='e'*64
    if fault=='policy':policy={'different':True}
    if fault=='flags':r['original_applications_modified']=True
    if fault=='source_file':(p['tmp_path']/r['source_pkml']['path']).write_text('changed')
    if fault=='evidence_file':(p['tmp_path']/'job/policy.json').write_text('changed')
    if fault=='code_file':(p['tmp_path']/p['native']).write_text('changed')
    if fault=='duplicate_code':p['write'](p['protocol_path'],{'code':p['code']+[p['code'][0]]})
    if fault=='unsupported_route':plan['adapter_route']='unregistered'
    with pytest.raises(ValueError):m._authenticate(p['f'],plan,r,policy)


def test_missing_retention_authentication_is_never_replaced_by_receipt_checks(fixture,monkeypatch):
    p=fixture
    def rejected(*args,**kwargs):raise ValueError('record is not uniquely registered for retention')
    monkeypatch.setattr(m,'binding_profile',rejected)
    with pytest.raises(ValueError,match='not uniquely registered'):
        m._authenticate(p['f'],p['plan'],p['receipt'],p['policy'])
