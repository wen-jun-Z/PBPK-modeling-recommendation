"""No live experiment processes are launched or signaled by these tests."""
import importlib.util
import json
from pathlib import Path
import signal
import sys

import pytest

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('suite_followup_under_test',SCRIPTS/'run_blueprint_followup.py')
follow=importlib.util.module_from_spec(spec)
spec.loader.exec_module(follow)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def process(proc,pid,ticks,*,ppid=1,state='S',children=(),threads=None):
    directory=proc/str(pid);directory.mkdir(parents=True,exist_ok=True)
    fields=['0']*22
    fields[0],fields[1],fields[11],fields[19],fields[21]=state,str(ppid),'10',str(ticks),'3'
    (directory/'stat').write_text(f'{pid} (owned worker)) '+' '.join(fields))
    for tid,values in {pid:children,**(threads or {})}.items():
        task=directory/'task'/str(tid);task.mkdir(parents=True,exist_ok=True)
        (task/'children').write_text(' '.join(map(str,values)))
    return {'pid':pid,'start_ticks':str(ticks)}


def suite(tmp_path,monkeypatch,*,finished=False):
    monkeypatch.setattr(follow,'ROOT',tmp_path)
    proc=tmp_path/'proc';root=process(proc,100,11,children=[200])
    process(proc,200,22,ppid=100,threads={201:[300]})
    process(proc,300,33,ppid=200)
    out=tmp_path/'experiments/prior'
    save(out/'configuration.json',{'version':'blueprint-reviewed-development-suite-v1','tasks':[{'task_id':'main__toy'}]})
    save(out/'runner.json',root)
    save(out/'queue.json',{'tasks':[{'task_id':'main__toy','status':'development_completed' if finished else 'running',
        'pid':200,'start_ticks':'22','output':'experiments/prior/tasks/main__toy'}]})
    save(out/'progress.json',{'status':'completed' if finished else 'running','stage':'finished' if finished else 'development',
        'active_tasks':[] if finished else ['main__toy']})
    if finished:save(out/'completion.json',{'status':'finished','task_status_counts':{'development_completed':1}})
    return follow.RegisteredSuite(out,follow.ProcessScope(proc_root=proc)),proc,out


def test_normal_running_suite_is_observed_without_signaling(tmp_path,monkeypatch):
    reg,_,_=suite(tmp_path,monkeypatch)
    monkeypatch.setattr(reg.scope,'send',lambda *args:pytest.fail('Must never signal prerequisite'))
    assert reg.sample() is False
    assert set(reg.known)=={100,200,300}


def test_completion_does_not_start_followup_before_all_thread_children_exit(tmp_path,monkeypatch):
    reg,proc,_=suite(tmp_path,monkeypatch,finished=True)
    assert reg.sample() is False
    process(proc,100,11,state='Z')
    process(proc,200,22,ppid=1,state='Z')
    process(proc,300,33,ppid=1)
    assert reg.sample() is False
    assert set(reg.known)=={300}
    process(proc,300,999,ppid=1)  # Reused PID is no longer this suite's process.
    assert reg.sample() is True


def test_monitor_can_supply_previously_tracked_owned_orphan(tmp_path,monkeypatch):
    reg,proc,out=suite(tmp_path,monkeypatch,finished=True)
    process(proc,100,11,state='Z');process(proc,200,22,state='Z')
    process(proc,300,33,ppid=1)
    save(out/'monitor.json',{'supervisor':reg.root,'tracked_processes':[{'pid':300,'start_ticks':'33'}]})
    assert reg.sample() is False
    assert set(reg.known)=={300}


def test_dead_supervisor_without_complete_is_not_treated_as_ready(tmp_path,monkeypatch):
    reg,proc,_=suite(tmp_path,monkeypatch)
    process(proc,100,11,state='Z')
    with pytest.raises(RuntimeError,match='without a verified complete'):reg.sample()


def test_completion_written_between_record_read_and_exit_check_is_retried_once(tmp_path,monkeypatch):
    reg,proc,out=suite(tmp_path,monkeypatch,finished=True)
    (out/'completion.json').unlink()
    for pid,ticks in ((100,11),(200,22),(300,33)):process(proc,pid,ticks,state='Z')
    real_state=reg.scope.state;reads=0
    def state(pid,ticks=None):
        nonlocal reads
        if pid==100:
            reads+=1
            if reads==2:save(out/'completion.json',{'status':'finished','task_status_counts':{'development_completed':1}})
        return real_state(pid,ticks)
    monkeypatch.setattr(reg.scope,'state',state)
    assert reg.sample() is True


def test_completion_published_after_old_queue_progress_reads_is_retried(tmp_path,monkeypatch):
    reg,proc,out=suite(tmp_path,monkeypatch)
    for pid,ticks in ((100,11),(200,22),(300,33)):process(proc,pid,ticks,state='Z')
    real_read=follow.read_json;changed=False
    def read(path):
        nonlocal changed
        value=real_read(path)
        if path==out/'progress.json' and not changed:
            changed=True
            save(out/'progress.json',{'status':'completed','stage':'finished','active_tasks':[]})
            save(out/'queue.json',{'tasks':[{'task_id':'main__toy','status':'development_completed'}]})
            save(out/'completion.json',{'status':'finished','task_status_counts':{'development_completed':1}})
        return value
    monkeypatch.setattr(follow,'read_json',read)
    assert reg.sample() is True


@pytest.mark.parametrize('damage',['queue','count','failed','monitor','configuration'])
def test_inconsistent_scope_or_failure_is_rejected(tmp_path,monkeypatch,damage):
    reg,_,out=suite(tmp_path,monkeypatch,finished=True)
    if damage=='queue':save(out/'queue.json',{'tasks':[{'task_id':'unregistered'}]})
    elif damage=='count':save(out/'completion.json',{'status':'finished','task_status_counts':{'development_completed':2}})
    elif damage=='failed':save(out/'queue.json',{'tasks':[{'task_id':'main__toy','status':'failed'}]})
    elif damage=='monitor':save(out/'monitor.json',{'supervisor':{'pid':999,'start_ticks':'999'},'tracked_processes':[]})
    else:save(out/'configuration.json',{'version':'changed'})
    with pytest.raises((ValueError,RuntimeError)):reg.sample()


def test_concurrency_includes_explicit_other_project_reservations():
    assert follow.parallelism(3,3,4,16,available=56)['declared_combined_cpu_upper_bound']==52
    assert follow.parallelism(2,2,2,44,available=56)['native_cpu_upper_bound']==8
    for args in [(4,4,4,0),(3,3,4,24),(0,3,4,0),(2,2,3,0),(2,2,2,-1)]:
        with pytest.raises(ValueError):follow.parallelism(*args,available=56)


def resources(tmp_path,monkeypatch):
    monkeypatch.setattr(follow,'ROOT',tmp_path)
    script=tmp_path/'scripts/run_blueprint_followup.py'
    monkeypatch.setattr(follow,'__file__',str(script))
    stage=tmp_path/'runtime/stage';qwen=tmp_path/'experiments/qwen';manifest=tmp_path/'experiments/cases.json'
    for path in [script,tmp_path/'scripts/watch_blueprint_suite.py',stage/'src/pbpk_kb/blueprints/design_runtime.py',
        stage/'src/pbpk_kb/another_imported_module.py',
        *(stage/'scripts'/n for n in ('run_blueprint_reviewed_suite.py','run_blueprint_experiments.py','run_blueprint_cohort.py')),
        *(tmp_path/'src/pbpk_kb/model_runtime/R'/n for n in ('blueprint_snapshot_run.R','blueprint_parameter_batch.R'))]:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# frozen source')
    bundle=tmp_path/'experiments/case.json';save(bundle,{'case_id':'toy'})
    save(manifest,{'cases':[{'case_id':'toy','bundle_path':str(bundle.relative_to(tmp_path)),'bundle_sha256':follow.checksum(bundle)}]})
    save(qwen/'progress.json',{'state':'completed','failures':0})
    hashes={}
    for seed in (17,43,101):
        path=qwen/'cases/toy'/f'seed_{seed}.json'
        save(path,{'case_id':'toy','seed':seed,'status':'completed','request_sha256':'request','weights_manifest_sha256':'weights'})
        hashes[str(path.relative_to(qwen))]=follow.checksum(path)
    save(qwen/'manifest.json',{'version':'pbpk-local-qwen-proposals-v2','final_status':'completed','requests':{'toy':'request'},
        'seeds':[17,43,101],'record_sha256':hashes,'weights_manifest_sha256':'weights'})
    return stage,manifest,qwen


def test_freeze_covers_seed_records_and_imported_sources_and_detects_mutation(tmp_path,monkeypatch):
    stage,manifest,qwen=resources(tmp_path,monkeypatch)
    frozen=follow.freeze_resources(stage,manifest,qwen)
    assert 'runtime/stage/src/pbpk_kb/another_imported_module.py' in frozen
    assert 'experiments/qwen/cases/toy/seed_101.json' in frozen
    follow.verify_frozen(frozen)
    (stage/'src/pbpk_kb/another_imported_module.py').write_text('# changed')
    with pytest.raises(RuntimeError,match='Frozen input changed'):follow.verify_frozen(frozen)


@pytest.mark.parametrize('damage',['record','request','weights','incomplete'])
def test_qwen_manifest_record_bindings_cannot_be_silently_refrozen(tmp_path,monkeypatch,damage):
    stage,manifest,qwen=resources(tmp_path,monkeypatch)
    path=qwen/'cases/toy/seed_17.json';record=json.loads(path.read_text())
    if damage=='record':record['rationale']='altered'
    elif damage=='request':record['request_sha256']='different'
    elif damage=='weights':record['weights_manifest_sha256']='different'
    else:record['status']='failed'
    save(path,record)
    with pytest.raises(ValueError):follow.freeze_resources(stage,manifest,qwen)


def test_error_cleanup_only_uses_child_tree_and_pidfd_checked_signaling(tmp_path,monkeypatch):
    reg,proc,_=suite(tmp_path,monkeypatch)
    process(proc,900,99)  # Unrelated predecessor identity never supplied to cleanup.
    known=reg.scope.discover([reg.root]);sent=[]
    monkeypatch.setattr(reg.scope,'send',lambda identity,sig:sent.append((identity['pid'],sig)) or True)
    result=follow.stop_owned(reg.scope,known,sleep=lambda _:None)
    assert {pid for pid,_ in sent}=={100,200,300}
    assert {sig for _,sig in sent}=={signal.SIGTERM,signal.SIGKILL}
    assert len(result['remaining_tracked_processes'])==3
