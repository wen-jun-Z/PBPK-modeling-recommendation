"""Historical producer hashes never waive a source or zero-invariance proof."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from recommendation_execution_repairs import scoring as s, coadministration_isolation as iso, retained_implementations as retained

@pytest.fixture
def case(tmp_path,monkeypatch):
    name='scripts/recommendation_execution_repairs/coadministration_isolation.py'
    before='1'*64;after='2'*64
    monkeypatch.setattr(iso,'IMPLEMENTATION_FILES',(name,))
    monkeypatch.setattr(s.complete_candidate,'IMPLEMENTATION_FILES',(name,))
    old={'implementation_sha256':{name:before},'target':'Drug',
         'source_reference':{'sha256':'a'*64},'events':[{'dose_mg':0,'new_start_time_min':1000}],
         'zero_manifold_proof':{'zero_monitor_paths':['metabolite'], 'positive_domain_monitor_dimensions':{'liver':'Volume'}},
         'original_isolation_policy_sha256':'c'*64}
    current=deepcopy(old);current['implementation_sha256'][name]=after
    plan={'implementation_sha256':{name:before},'scenario_id':'s','candidate_id':'c'}
    protocol={'code':[{'path':name,'sha256':after}]}
    f=SimpleNamespace(root=tmp_path,read=lambda p:protocol,checked=lambda r:tmp_path/r['path'])
    calls=[]
    def binding(record,frozen,**kw):
        calls.append((record,frozen,kw));return {name:before},{name}
    monkeypatch.setattr(retained,'binding_profile',binding)
    return SimpleNamespace(name=name,before=before,after=after,old=old,current=current,plan=plan,f=f,calls=calls)


def test_unchanged_old_policy_bound_to_authenticated_original_producer(case):
    c=case;original=deepcopy(c.old)
    assert s.bind_rederived_isolation_policy(c.f,c.plan,c.old,c.current) is c.old
    assert c.old==original and len(c.calls)==1
    assert c.calls[0][2]['adapter_path']=='scripts/recommendation_execution_repairs/complete_candidate.py'
    assert c.calls[0][0] is c.plan


@pytest.mark.parametrize('change',['dose','time','source','target','zero_monitor','domain','original_digest'])
def test_semantic_difference_never_becomes_hash_compatibility(case,change):
    c=case;fresh=deepcopy(c.current)
    if change=='dose':fresh['events'][0]['dose_mg']=1
    elif change=='time':fresh['events'][0]['new_start_time_min']=0
    elif change=='source':fresh['source_reference']['sha256']='b'*64
    elif change=='target':fresh['target']='Different'
    elif change=='zero_monitor':fresh['zero_manifold_proof']['zero_monitor_paths']=[]
    elif change=='domain':fresh['zero_manifold_proof']['positive_domain_monitor_dimensions']={}
    else:fresh['original_isolation_policy_sha256']='d'*64
    with pytest.raises(ValueError,match='scientific proof differs'):
        s.bind_rederived_isolation_policy(c.f,c.plan,c.old,fresh)
    assert not c.calls


@pytest.mark.parametrize('change',['old_hash','fresh_hash','plan_hash','missing_dependency','additional_dependency'])
def test_producer_substitution_rejected(case,change):
    c=case
    if change=='old_hash':c.old['implementation_sha256'][c.name]='3'*64
    elif change=='fresh_hash':c.current['implementation_sha256'][c.name]='3'*64
    elif change=='plan_hash':c.plan['implementation_sha256'][c.name]='3'*64
    elif change=='missing_dependency':c.old['implementation_sha256']={}
    else:c.old['implementation_sha256']['unexpected.py']='3'*64
    with pytest.raises(ValueError):s.bind_rederived_isolation_policy(c.f,c.plan,c.old,c.current)


def test_unregistered_plan_is_rejected(case,monkeypatch):
    def reject(*a,**kw):raise ValueError('unregistered original record')
    monkeypatch.setattr(retained,'binding_profile',reject)
    with pytest.raises(ValueError,match='unregistered original record'):
        s.bind_rederived_isolation_policy(case.f,case.plan,case.old,case.current)


def test_current_code_file_drift_is_rejected(case):
    def reject(ref):raise ValueError('actual frozen code changed')
    case.f.checked=reject
    with pytest.raises(ValueError,match='actual frozen code changed'):
        s.bind_rederived_isolation_policy(case.f,case.plan,case.old,case.current)
