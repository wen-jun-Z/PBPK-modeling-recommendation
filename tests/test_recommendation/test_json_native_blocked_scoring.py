"""Blocked bridge records cannot use their dependency marker to smuggle predictions."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import types
import pytest

STAGED=next((p for p in Path(__file__).resolve().parents if p.name=='staged'),None)
PROJECT=Path('/home/xuyiqian/Research_Projects/Knowledge_Graph_Model_Recommendation')
STAGED=STAGED or PROJECT
sys.path[:0]=[str(PROJECT/'scripts'),str(PROJECT/'src')]
spec=importlib.util.spec_from_file_location('blocked_bridge_scoring',STAGED/'scripts/recommendation_execution_repairs/scoring.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


@pytest.fixture
def binding(monkeypatch):
    import recommendation_execution_repairs as package
    spec=importlib.util.spec_from_file_location('blocked_bridge_retention',STAGED/'scripts/recommendation_execution_repairs/retained_implementations.py')
    retained=importlib.util.module_from_spec(spec);spec.loader.exec_module(retained)
    monkeypatch.setitem(sys.modules,'recommendation_execution_repairs.retained_implementations',retained)
    bridge=types.ModuleType('recommendation_execution_repairs.json_native_candidate')
    bridge.VERSION='source-bound-json-native-candidate-v1'
    bridge.IMPLEMENTATION_FILES=tuple(set(s.snapshot_candidate.IMPLEMENTATION_FILES)|set(s.complete_candidate.IMPLEMENTATION_FILES)|{
        'scripts/recommendation_execution_repairs/json_native_candidate.py',
        'scripts/recommendation_execution_repairs/json_native_source_proof.py'})
    monkeypatch.setitem(sys.modules,bridge.__name__,bridge)
    monkeypatch.setattr(package,'json_native_candidate',bridge,raising=False)
    code={p:'a'*64 for p in bridge.IMPLEMENTATION_FILES}
    record={'adapter_route':'blocked_json_native_bridge','version':bridge.VERSION,'status':'blocked',
            'execution_revision':s.REVISION,'implementation_sha256':deepcopy(code)}
    def verify(value):
        with s.repaired_execution_profile():
            return s.backend.verify_execution_binding(value,frozen_code_sha256=code,
                expected_revision=s.REVISION,label='blocked bridge')
    return record,code,verify


@pytest.mark.parametrize('result_flags',[False,True])
def test_only_unexecuted_blocked_plan_or_result_authenticates(binding,result_flags):
    record,code,verify=binding
    if result_flags:record.update(native_executed=False,prediction_locked=False)
    assert verify(record)==code


@pytest.mark.parametrize('fault',['ready','prediction_status','version','native_true','locked_true',
    'native_zero','locked_zero','prediction_file','prediction_lock','predicted_endpoints','route','missing_code','code_hash',
    'missing_bridge_marker','missing_marker_prediction'])
def test_blocked_profile_refuses_predictions_wrong_routes_and_incomplete_binding(binding,fault):
    record,code,verify=binding
    if fault=='ready':record['status']='ready'
    elif fault=='prediction_status':record['status']='prediction_locked'
    elif fault=='version':record['version']='snapshot-v1'
    elif fault=='native_true':record['native_executed']=True
    elif fault=='locked_true':record['prediction_locked']=True
    elif fault=='native_zero':record['native_executed']=0
    elif fault=='locked_zero':record['prediction_locked']=0
    elif fault in {'prediction_file','prediction_lock','predicted_endpoints'}:record[fault]=None
    elif fault=='route':record['adapter_route']='complete_original_pksim5'
    elif fault=='missing_code':record['implementation_sha256'].pop('scripts/recommendation_execution_repairs/json_native_source_proof.py')
    elif fault in {'missing_bridge_marker','missing_marker_prediction'}:
        record['implementation_sha256'].pop('scripts/recommendation_execution_repairs/json_native_candidate.py')
        if fault=='missing_marker_prediction':record.update(status='prediction_locked',prediction_locked=True)
    else:record['implementation_sha256']['scripts/recommendation_execution_repairs/json_native_source_proof.py']='b'*64
    with pytest.raises((ValueError,FileNotFoundError)):verify(record)
