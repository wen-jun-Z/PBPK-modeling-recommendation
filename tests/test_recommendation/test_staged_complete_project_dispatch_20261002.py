"""Early native dispatch uses a separate immutable plan and the final lock gates."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT/'scripts/recommendation_execution_repairs/pipeline.py'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def phase(monkeypatch):
    m = types.ModuleType('staged_early_pipeline')
    m.__file__ = str(LIVE)
    exec(compile(LIVE.read_text(), str(LIVE), 'exec'), m.__dict__)
    scratch = ROOT/'runtime/tmp'; scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='early-pipeline-', dir=scratch) as directory:
        base = Path(directory)
        for name in ('OUT', 'WORK', 'RUN'):
            path = base/name.lower(); path.mkdir(); monkeypatch.setattr(m, name, path)
        monkeypatch.setattr(m, 'STOP', m.RUN/'stop.requested')
        # Synthetic dispatch tests use synthetic identities, independent of the real continuation registry.
        monkeypatch.setattr(m, 'job_key', lambda row: m.original.pair_key(row).removesuffix('_r2')+'_fixture')
        audits=[]; scenarios={}; jobs=[]; bindings={}
        for i in range(64):
            row={'scenario_id':f's{i}', 'candidate_id':f'c{i}', 'identity_match':True,
                 'catalog_provenance':{'source_path':f'runtime/tmp/model_{i}.pksim5', 'source_sha256':f'{i:064x}'}}
            plan={'scenario_id':row['scenario_id'], 'candidate_id':row['candidate_id'],
                  'adapter_route':'complete_original_pksim5', 'execution_revision':m.REVISION,
                  'status':'blocked', 'implementation_sha256':{}, 'reasons':['Blocked:SYNTHETIC']}
            # Early design blockers legitimately precede source_path assignment.
            job={'job_id':m.job_key(row), 'scenario_id':row['scenario_id'], 'candidate_id':row['candidate_id'], 'plan':plan}
            audits.append(row); jobs.append(job); scenarios[row['scenario_id']]={'scenario_id':row['scenario_id']}
            bindings[job['job_id']]={'path':row['catalog_provenance']['source_path'], 'sha256':row['catalog_provenance']['source_sha256']}
        audit_path=m.OUT/'inputs/candidate_audit.jsonl'; audit_path.parent.mkdir(parents=True)
        audit_path.write_text(''.join(json.dumps(r)+'\n' for r in audits))
        (m.OUT/'inputs/scenarios.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scenarios.values()))
        record={'execution_revision':m.REVISION, 'jobs':jobs, 'source_bindings':bindings,
                'candidate_audit':m.reference(audit_path), 'all_complete_project_candidates_included':True,
                'selected_using_rank_or_error':False, 'sealed_outcomes_accessed':False}
        early=m.WORK/'complete_project_execution_plan.json'; write(early, record)
        controller=object.__new__(m.Controller)
        controller.guard=lambda: None
        write(m.WORK/'preparation.json', {'runtime':{'native_timeout_seconds':600,'complete_project_native_timeout_seconds':1800}})
        monkeypatch.setattr(m, 'freeze', lambda: {'runtime':{'native_timeout_seconds':600,'complete_project_native_timeout_seconds':1800}})
        yield types.SimpleNamespace(m=m, base=base, audits=audits, scenarios=scenarios, jobs=jobs,
                                    record=record, early=early, controller=controller)


def test_all_64_including_source_unassigned_blockers_are_registered(phase):
    assert phase.m.verify_complete_project_plan(phase.record, phase.audits)==phase.jobs


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'json_source', 'wrong_binding', 'wrong_plan_source', 'wrong_identity', 'wrong_route'])
def test_complete_phase_rejects_scope_and_source_substitution(phase, change):
    f=phase; record=deepcopy(f.record); audits=deepcopy(f.audits)
    if change=='missing':record['jobs'].pop()
    elif change=='duplicate':record['jobs'][-1]=deepcopy(record['jobs'][0])
    elif change=='json_source':audits[0]['catalog_provenance']['source_path']='runtime/tmp/other.json'
    elif change=='wrong_binding':record['source_bindings'][f.jobs[0]['job_id']]['sha256']='f'*64
    elif change=='wrong_plan_source':record['jobs'][0]['plan']['source_path']='runtime/tmp/substitute.json'
    elif change=='wrong_identity':record['jobs'][0]['plan']['candidate_id']='other'
    elif change=='wrong_route':record['jobs'][0]['plan']['adapter_route']='snapshot'
    with pytest.raises(ValueError):f.m.verify_complete_project_plan(record,audits)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'contract', 'implementation', 'identity'])
def test_final_full_plan_requires_exact_unique_early_subset(phase, change):
    f=phase; jobs=deepcopy(f.jobs)
    if change=='missing':jobs.pop()
    elif change=='duplicate':jobs.append(deepcopy(jobs[0]))
    elif change=='identity':jobs[0]['scenario_id']='other'
    else:jobs[0]['plan'][change]={'changed':True}
    with pytest.raises(ValueError, match='Final candidate plan differs'):
        f.m.verify_complete_project_subset(jobs,f.jobs)


def test_early_worker_uses_fixed_plan_and_passes_frozen_plan(phase,monkeypatch):
    f=phase;m=f.m;job=f.jobs[0]; calls=[]
    # An unrelated final plan must never be selected by the early worker.
    altered=deepcopy(job);altered['plan']['status']='changed'
    write(m.WORK/'execution_plan.json', {'jobs':[altered]})
    executor=types.SimpleNamespace(prepare_candidate=lambda *a,**kw:deepcopy(job['plan']),
        execute_candidate=lambda *a,**kw:calls.append((a,kw)))
    monkeypatch.setattr(m,'adapter',lambda _:executor)
    monkeypatch.setattr(m.signal,'signal',lambda *a:None)
    checked=[];monkeypatch.setattr(m,'verify_job',lambda j:checked.append(j))
    m.worker(job['job_id'],'complete_projects')
    args,kw=calls[0]
    assert args[2]==m.OUT/'native'/job['job_id']
    assert kw['frozen_plan']==job['plan'] and kw['cpu_threads']==1 and kw['stop_file']==m.STOP
    assert kw['timeout_seconds']==1800
    assert checked==[job]
    with pytest.raises(ValueError,match='Unknown registered worker stage'):m.worker(job['job_id'],'../other')


def test_child_passes_stage_to_owned_worker(phase,monkeypatch):
    f=phase;m=f.m;commands=[]
    process=types.SimpleNamespace(pid=999999,returncode=0,poll=lambda:0)
    monkeypatch.setattr(m.subprocess,'Popen',lambda command,**kw:commands.append(command) or process)
    monkeypatch.setattr(m.original,'identity',lambda _: {'pid':999999})
    monkeypatch.setattr(m,'verify_job',lambda _:None)
    f.controller.lock=threading.Lock();f.controller.children={};f.controller.limits={'worker_timeout_seconds':900}
    f.controller.child(f.jobs[0],stage='complete_projects')
    assert commands[0][-2:]==['--stage','complete_projects']


def test_early_dispatch_is_bounded_immutable_and_preserves_every_job(phase,monkeypatch):
    f=phase;m=f.m;lock=threading.Lock();active=0;peak=0;seen=[];verified=[]
    def child(job,stage):
        nonlocal active,peak
        assert stage=='complete_projects'
        with lock:active+=1;peak=max(peak,active);seen.append(job['job_id'])
        time.sleep(.002)
        with lock:active-=1
    monkeypatch.setattr(f.controller,'child',child)
    monkeypatch.setattr(m,'verify_job',lambda j:verified.append(j['job_id']))
    before=f.early.read_bytes()
    assert f.controller.run_complete_projects(f.audits,f.scenarios)==f.jobs
    assert peak==4 and len(seen)==64 and len(set(seen))==64
    assert verified==[j['job_id'] for j in f.jobs] and f.early.read_bytes()==before
    assert not (m.WORK/'execution_plan.json').exists()


def test_existing_final_plan_is_reconciled_before_any_early_dispatch(phase,monkeypatch):
    f=phase;m=f.m;jobs=deepcopy(f.jobs);jobs[0]['plan']['status']='changed'
    write(m.WORK/'execution_plan.json',{'jobs':jobs})
    monkeypatch.setattr(f.controller,'child',lambda *a,**kw:pytest.fail('must not dispatch mismatched plan'))
    with pytest.raises(ValueError,match='Final candidate plan differs'):
        f.controller.run_complete_projects(f.audits,f.scenarios)


def test_existing_terminal_result_is_verified_then_skipped_and_partial_preserved(phase,monkeypatch):
    f=phase;m=f.m;job=f.jobs[0];directory=m.OUT/'native'/job['job_id'];directory.mkdir(parents=True)
    plan_path=directory/'execution_plan.json';write(plan_path,job['plan'])
    result={'status':'blocked','prediction_locked':False,'scenario_id':job['scenario_id'],'candidate_id':job['candidate_id'],
            'execution_revision':m.REVISION,'implementation_sha256':{},'execution_plan_sha256':m.sha(plan_path),
            'sealed_observations_accessed':False,'fitting_performed':False}
    write(directory/'result.json',result)
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**kw:pytest.fail('must reuse strictly verified result'))
    f.controller.child(job,stage='complete_projects')
    result['execution_plan_sha256']='0'*64;write(directory/'result.json',result)
    with pytest.raises(ValueError,match='Frozen native job differs'):f.controller.child(job,stage='complete_projects')
    (directory/'result.json').unlink()
    with pytest.raises(RuntimeError,match='Preserve incomplete job'):f.controller.child(job,stage='complete_projects')
    assert plan_path.exists()


def test_controller_dispatches_early_before_waiting_for_sources(phase,monkeypatch):
    f=phase;order=[]
    monkeypatch.setattr(f.controller,'run_complete_projects',lambda *a:order.append('early') or f.jobs)
    def wait():
        order.append('await_sources');raise InterruptedError('synthetic source wait')
    monkeypatch.setattr(f.controller,'await_existing_source_repairs',wait)
    monkeypatch.setattr(f.controller,'score_locked',lambda *a:pytest.fail('early phase cannot score'))
    with pytest.raises(InterruptedError,match='synthetic source wait'):f.controller.run()
    assert order==['early','await_sources']


def test_full_177_run_reuses_early_64_and_still_rejects_unverified_source(phase,monkeypatch):
    f=phase;m=f.m;audits=deepcopy(f.audits);scenarios=deepcopy(f.scenarios)
    plans={(j['scenario_id'],j['candidate_id']):j['plan'] for j in f.jobs}
    for i in range(64,177):
        row={'scenario_id':f's{i}','candidate_id':f'c{i}','identity_match':True,
             'catalog_provenance':{'source_path':f'runtime/tmp/model_{i}.json','source_sha256':f'{i:064x}'}}
        audits.append(row);scenarios[row['scenario_id']]={'scenario_id':row['scenario_id']}
        plans[(row['scenario_id'],row['candidate_id'])]={
            'scenario_id':row['scenario_id'],'candidate_id':row['candidate_id'],'status':'blocked',
            'reasons':['Blocked:SYNTHETIC'],'execution_revision':m.REVISION,'implementation_sha256':{}}
    audit_path=m.OUT/'inputs/candidate_audit.jsonl'
    audit_path.write_text(''.join(json.dumps(r)+'\n' for r in audits))
    (m.OUT/'inputs/scenarios.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scenarios.values()))
    f.record['candidate_audit']=m.reference(audit_path);write(f.early,f.record)
    executor=types.SimpleNamespace(prepare_candidate=lambda scenario,row,**kw:deepcopy(plans[(row['scenario_id'],row['candidate_id'])]))
    monkeypatch.setattr(m,'adapter',lambda _:executor)
    events=[];created=[];lock=threading.Lock()
    def child(job,stage='all'):
        directory=m.OUT/'native'/job['job_id']
        if (directory/'result.json').exists():m.verify_job(job);return
        directory.mkdir(parents=True)
        plan_path=directory/'execution_plan.json';write(plan_path,job['plan'])
        write(directory/'result.json',{'status':'blocked','prediction_locked':False,
            'scenario_id':job['scenario_id'],'candidate_id':job['candidate_id'],
            'execution_revision':m.REVISION,'implementation_sha256':{},
            'execution_plan_sha256':m.sha(plan_path),'sealed_observations_accessed':False,
            'fitting_performed':False,'reasons':['Blocked:SYNTHETIC']})
        with lock:created.append(job['job_id']);events.append(stage)
    monkeypatch.setattr(f.controller,'child',child)
    monkeypatch.setattr(f.controller,'await_existing_source_repairs',lambda:events.append('await_sources'))
    monkeypatch.setattr(f.controller,'score_locked',lambda *a:pytest.fail('unverified source cannot score'))
    from recommendation_execution_repairs import snapshot_repair,engineering_gate
    source={'status':'complete','all_sources_accounted':True,'expected_sources':81,'verified_sources':80,
            'results':[{'status':'failed','source_sha256':'unknown','reasons':['author data absent']}]}
    monkeypatch.setattr(snapshot_repair,'prepare_repairs',lambda *a,**kw:source)
    # Isolate the still-strict source gate from synthetic blocked-plan context.
    monkeypatch.setattr(engineering_gate,'assess_blocked_plan',lambda _: {'accepted':True,'scientific_applicability':'unknown'})
    f.controller.limits={'workers':4,'source_stage_timeout_seconds':600}
    f.controller.run()
    assert len(created)==177 and len(set(created))==177
    assert events[0]=='await_sources'
    assert events[1:65]==['complete_projects']*64 and events[65]=='await_sources'
    final=json.loads((m.WORK/'execution_plan.json').read_text())
    assert len(final['jobs'])==177
    prediction_lock=json.loads((m.WORK/'predictions_lock.json').read_text())
    assert prediction_lock['planned_jobs']==177 and prediction_lock['target_observations_read'] is False
    gate=json.loads((m.WORK/'engineering_completeness.json').read_text())
    assert gate['accepted'] is False and any(x['stage']=='source_repair' for x in gate['issues'])
    assert json.loads((m.RUN/'status.json').read_text())['state']=='engineering_repairs_required'


def test_first_early_plan_creation_freezes_all_64_without_overwrite(phase,monkeypatch):
    f=phase;m=f.m;f.early.unlink();calls=[]
    plans={(j['scenario_id'],j['candidate_id']):j['plan'] for j in f.jobs}
    def prepare(scenario,row,**kw):
        calls.append((row['scenario_id'],row['candidate_id']))
        return deepcopy(plans[calls[-1]])
    monkeypatch.setattr(m,'adapter',lambda _:types.SimpleNamespace(prepare_candidate=prepare))
    monkeypatch.setattr(f.controller,'child',lambda *a,**kw:None)
    monkeypatch.setattr(m,'verify_job',lambda _:None)
    assert f.controller.run_complete_projects(f.audits,f.scenarios)==f.jobs
    frozen=f.early.read_bytes()
    assert len(calls)==64 and json.loads(frozen)==f.record
    f.controller.run_complete_projects(f.audits,f.scenarios)
    assert len(calls)==64 and f.early.read_bytes()==frozen


def test_stop_request_prevents_all_early_worker_launches(phase,monkeypatch):
    f=phase;m=f.m;m.STOP.touch()
    f.controller.guard=types.MethodType(m.Controller.guard,f.controller)
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**kw:pytest.fail('stop must prevent worker launch'))
    with pytest.raises(InterruptedError,match='Project repair stop requested'):
        f.controller.run_complete_projects(f.audits,f.scenarios)
    assert not (m.OUT/'native').exists()


def test_registered_full_plan_returns_to_shared_pool_without_early_dispatch(phase,monkeypatch):
    f=phase
    write(f.m.WORK/'execution_plan.json',{'jobs':f.jobs})
    monkeypatch.setattr(f.controller,'child',lambda *a,**kw:pytest.fail('full-plan resume belongs to shared pool'))
    monkeypatch.setattr(f.controller,'await_existing_source_repairs',lambda:pytest.fail('source wait happens in main run'))
    assert f.controller.run_complete_projects(f.audits,f.scenarios)==f.jobs


def test_real_continuation_preserves_all_177_pairs_and_unique_attempts():
    from recommendation_execution_repairs import pipeline as p,retained_implementations as h
    manifest=json.loads((ROOT/h.MANIFEST).read_text())
    mapping=manifest['job_mapping']
    assert len(mapping)==177 and len({r['job_id'] for r in mapping})==177
    assert 0 < sum(r['retained'] for r in mapping) < 177
    audits={(r['scenario_id'],r['candidate_id']):r for r in p.rows(p.OUT/'inputs/candidate_audit.jsonl') if r.get('identity_match') is True}
    assert len(audits)==177
    for row in mapping:
        assert p.job_key(audits[(row['scenario_id'],row['candidate_id'])])==row['job_id']
    with pytest.raises(ValueError,match='not registered'):
        p.job_key({'scenario_id':'unregistered','candidate_id':'unregistered'})


@pytest.mark.parametrize("route,expected", [("complete_original_pksim5",1800),("verified_json_native_pksim5",1800),("original_snapshot",600)])
def test_shared_queue_uses_adapter_timeout_and_stop_guard(phase,monkeypatch,route,expected):
    f=phase;m=f.m;job=deepcopy(f.jobs[0]);job["plan"]["adapter_route"]=route
    write(m.WORK/"execution_plan.json",{"jobs":[job]})
    calls=[]
    executor=types.SimpleNamespace(prepare_candidate=lambda *a,**kw:deepcopy(job["plan"]),
        execute_candidate=lambda *a,**kw:calls.append(kw))
    monkeypatch.setattr(m,"adapter",lambda _:executor)
    monkeypatch.setattr(m.signal,"signal",lambda *a:None)
    monkeypatch.setattr(m,"verify_job",lambda _:None)
    m.worker(job["job_id"],"all")
    assert len(calls)==1 and calls[0]["timeout_seconds"]==expected
    if route in {"complete_original_pksim5","verified_json_native_pksim5"}:
        assert calls[0]["stop_file"]==m.STOP and calls[0]["frozen_plan"]==job["plan"]
        assert calls[0]["cpu_threads"]==1
    else:
        assert "frozen_plan" not in calls[0]


@pytest.mark.parametrize("alteration",["suffix_escape","unknown_previous","incorrect_new_id","wrong_original_pair"])
def test_continuation_rejects_invalid_registered_ids(monkeypatch,tmp_path,alteration):
    from recommendation_execution_repairs import pipeline as p,retained_implementations as h
    data=json.loads((ROOT/h.MANIFEST).read_text());row=next(r for r in data['job_mapping'] if not r['retained'])
    if alteration=='suffix_escape':data['continuation_suffix']='../../escape'
    elif alteration=='unknown_previous':row['previous_job_id']='other_r4b'
    elif alteration=='incorrect_new_id':row['job_id']=row['previous_job_id']
    else:row['old_job_id']='other_r4b'
    fixture=tmp_path/'registry.json';fixture.write_text(json.dumps(data));monkeypatch.setattr(p,'check',lambda _:fixture)
    with pytest.raises(ValueError):p.job_key(row)


def test_duplicate_proof_inputs_preserved_once_and_conflicts_rejected(phase):
    a={'path':'source.json','sha256':'a'*64};b={'path':'receipt.json','sha256':'b'*64}
    assert phase.m.unique_refs([a,b,dict(a)])==[a,b]
    with pytest.raises(ValueError,match='Conflicting frozen input binding'):
        phase.m.unique_refs([a,{'path':'source.json','sha256':'c'*64}])
