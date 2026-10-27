"""A blocked native bridge preserves its exact prepared evidence without any R."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.recommendation_execution_repairs import json_native_candidate as bridge


@pytest.fixture
def blocked(tmp_path, monkeypatch):
    scenario = {'scenario_id':'scenario', 'dose':200}
    candidate = {'candidate_id':'candidate', 'provenance':{'source_path':'source.json','source_sha256':'a'*64}}
    fallback = {'version':'snapshot-v1','status':'blocked', 'scenario_id':'scenario', 'candidate_id':'candidate',
        'execution_revision':'engineering_repair_04','implementation_sha256':{'original.py':'old'},
        'reasons':['Blocked:NO_SOURCE_SIMULATION_MATCHES_FIXED_EXECUTION_CONTRACT'],
        'sealed_observations_accessed':False, 'fitting_performed':False,
        'scientific_inapplicability_inferred':False, 'snapshot_evidence':{'unchanged':[1,2,3]}}
    entry = {'native_source':{'path':'native.pksim5','sha256':'b'*64}}
    proof = {'candidates':[entry], 'source_original':candidate['provenance'],
             'native_coverage_complete':True,'new_native_build_required':False,
             'rejected_native_artifacts':[]}
    hashes = {p:'c'*64 for p in bridge.IMPLEMENTATION_FILES}
    monkeypatch.setattr(bridge.snapshot_candidate,'prepare_candidate',lambda *a,**kw:deepcopy(fallback))
    monkeypatch.setattr(bridge.json_native_source_proof,'resolve_native_sources',lambda *a,**kw:deepcopy(proof))
    monkeypatch.setattr(bridge.complete_candidate,'prepare_candidate',lambda *a,**kw:{'status':'blocked',
        'reasons':['SOURCE_RELEASE_CONFIGURATION_GAP'], 'source_simulation':'Jalava_Treatment',
        'application_expansion_assessment_failures':[{'reason':'missing capsule release'}]})
    def implementation(root, *, verified_json_source=False):
        assert root == tmp_path and verified_json_source is True
        return deepcopy(hashes)
    monkeypatch.setattr(bridge.complete_candidate,'implementation_hashes',implementation)
    def never(*args,**kwargs):
        raise AssertionError('No snapshot, native or complete execution for a blocked bridge')
    monkeypatch.setattr(bridge.snapshot_candidate,'execute_candidate',never)
    monkeypatch.setattr(bridge.snapshot_candidate,'execute_snapshot',never)
    monkeypatch.setattr(bridge.complete_candidate,'execute_candidate',never)
    monkeypatch.setattr(bridge.shutil,'disk_usage',lambda path:SimpleNamespace(free=3*1024**3))
    plan = bridge.prepare_candidate(scenario,candidate,project_root=tmp_path)
    return SimpleNamespace(root=tmp_path,scenario=scenario,candidate=candidate,fallback=fallback,
                           proof=proof,hashes=hashes,plan=plan,output=tmp_path/'job')


def execute(p, **kwargs):
    options = dict(project_root=p.root, timeout_seconds=1800, frozen_plan=p.plan)
    options.update(kwargs)
    return bridge.execute_candidate(p.scenario,p.candidate,p.output,**options)


def test_blocked_bridge_saves_complete_unchanged_plan_without_native(blocked):
    p=blocked; original=deepcopy(p.plan)
    result=execute(p)
    saved=json.loads((p.output/'execution_plan.json').read_text())
    assert saved==original==p.plan
    assert saved['version']==bridge.VERSION and saved['adapter_route']==bridge.BLOCKED_ROUTE
    assert saved['implementation_sha256']==p.hashes and set(p.hashes)==set(bridge.IMPLEMENTATION_FILES)
    assert saved['snapshot_evidence']==p.fallback['snapshot_evidence']
    assert saved['json_native_bridge_assessment']['native_candidates'][0]['application_expansion_assessment_failures']==[{'reason':'missing capsule release'}]
    assert result==json.loads((p.output/'result.json').read_text())
    assert result['version']==saved['version'] and result['adapter_route']==saved['adapter_route']
    assert result['status']=='blocked' and result['reasons']==saved['reasons']
    assert result['execution_plan_sha256']==bridge.complete_candidate.sha(p.output/'execution_plan.json')
    assert all(result[k] is False for k in ('native_executed','prediction_locked','sealed_observations_accessed',
                                           'fitting_performed','scientific_inapplicability_inferred'))
    assert {f.name for f in p.output.iterdir()}=={'execution_plan.json','result.json'}
    assert p.fallback['version']=='snapshot-v1' and 'json_native_bridge_assessment' not in p.fallback


@pytest.mark.parametrize('change',['assessment','identity','version','route','implementation','reasons'])
def test_frozen_plan_tamper_is_rejected_before_writing(blocked,change):
    p=blocked;bad=deepcopy(p.plan)
    if change=='assessment':bad['json_native_bridge_assessment']['native_candidates']=[]
    elif change=='identity':bad['candidate_id']='different'
    elif change=='version':bad['version']='different'
    elif change=='route':bad['adapter_route']='verified_json_native_pksim5'
    elif change=='implementation':bad['implementation_sha256'].pop(next(iter(bad['implementation_sha256'])))
    else:bad['reasons']=['other']
    with pytest.raises(ValueError,match='Frozen JSON native bridge plan changed'):execute(p,frozen_plan=bad)
    assert not p.output.exists()


@pytest.mark.parametrize('fault',['missing_coverage','source_failure'])
def test_proof_resolution_failure_is_also_a_complete_blocked_bridge(blocked,monkeypatch,fault):
    p=blocked
    if fault=='missing_coverage':p.proof.update(native_coverage_complete=False,unbuilt_simulations=['unbuilt'])
    else:
        def failed(*args,**kwargs):raise ValueError('source hash changed')
        monkeypatch.setattr(bridge.json_native_source_proof,'resolve_native_sources',failed)
    p.plan=bridge.prepare_candidate(p.scenario,p.candidate,project_root=p.root)
    assert p.plan['adapter_route']==bridge.BLOCKED_ROUTE and p.plan['implementation_sha256']==p.hashes
    assert 'reason' in p.plan['json_native_bridge_assessment']
    assert execute(p)['status']=='blocked'


def test_source_assessment_change_after_freeze_is_rejected(blocked):
    p=blocked;p.proof['rejected_native_artifacts']=[{'reason':'changed source proof'}]
    with pytest.raises(ValueError,match='Frozen JSON native bridge plan changed'):execute(p)
    assert not p.output.exists()


def test_existing_attempt_is_never_overwritten(blocked):
    p=blocked;execute(p);before={x.name:x.read_bytes() for x in p.output.iterdir()}
    with pytest.raises(FileExistsError):execute(p)
    assert {x.name:x.read_bytes() for x in p.output.iterdir()}==before


@pytest.mark.parametrize('fault',['outside','sealed','root','symlink'])
def test_project_and_unsealed_output_boundary(blocked,fault):
    p=blocked
    if fault=='outside':p.output=p.root.parent/'forbidden-job'
    elif fault=='sealed':p.output=p.root/'sealed_outcomes'/'job'
    elif fault=='root':p.output=p.root
    else:
        (p.root/'outside-link').symlink_to(p.root.parent,target_is_directory=True)
        p.output=p.root/'outside-link'/'forbidden-job'
    with pytest.raises(ValueError,match='inside this project'):execute(p)


def test_disk_limit_precedes_any_record_write(blocked,monkeypatch):
    p=blocked;monkeypatch.setattr(bridge.shutil,'disk_usage',lambda path:SimpleNamespace(free=2*1024**3-1))
    with pytest.raises(OSError,match='2 GiB'):execute(p)
    assert not p.output.exists()


@pytest.mark.parametrize('timeout',[0,1801,True,1.5])
def test_blocked_route_preserves_timeout_bounds(blocked,timeout):
    with pytest.raises(ValueError,match='timeout_seconds'):execute(blocked,timeout_seconds=timeout)
    assert not blocked.output.exists()


def test_stop_request_preserves_empty_output_boundary(blocked):
    p=blocked;stop=p.root/'stop.requested';stop.write_text('stop')
    with pytest.raises(ValueError,match='PROJECT_STOP_REQUESTED'):execute(p,stop_file=stop)
    assert not p.output.exists()


def test_implementation_change_during_recording_cannot_publish_terminal_result(blocked,monkeypatch):
    p=blocked;old=bridge.snapshot_candidate._save
    def save(path,value):
        old(path,value)
        if path.name=='execution_plan.json':p.hashes[next(iter(p.hashes))]='d'*64
    monkeypatch.setattr(bridge.snapshot_candidate,'_save',save)
    with pytest.raises(ValueError,match='implementation changed during recording'):execute(p)
    assert (p.output/'execution_plan.json').exists() and not (p.output/'result.json').exists()


@pytest.mark.parametrize('status',['ready','blocked'])
def test_unattempted_snapshot_path_remains_unchanged(blocked,monkeypatch,status):
    p=blocked;fallback={'status':status,'reasons':['Blocked:CANDIDATE_LICENSE_NOT_RESOLVED']}
    monkeypatch.setattr(bridge.snapshot_candidate,'prepare_candidate',lambda *a,**kw:fallback)
    def untouched(*args,**kwargs):
        assert kwargs['timeout_seconds']==600
        return {'original_snapshot_execution':status}
    monkeypatch.setattr(bridge.snapshot_candidate,'execute_candidate',untouched)
    assert bridge.prepare_candidate(p.scenario,p.candidate,project_root=p.root) is fallback
    assert execute(p,frozen_plan=fallback)=={'original_snapshot_execution':status}
