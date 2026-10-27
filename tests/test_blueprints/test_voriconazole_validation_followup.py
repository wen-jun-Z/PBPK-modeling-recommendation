"""Synthetic registered processes only; no real task or outcome is touched."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('vori_followup_test',ROOT/'scripts/run_voriconazole_validation_followup.py')
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def process(proc,pid,ticks,*,ppid=1,state='S',children=(),threads=None):
    directory=proc/str(pid);directory.mkdir(parents=True,exist_ok=True);fields=['0']*22
    fields[0],fields[1],fields[11],fields[19],fields[21]=state,str(ppid),'10',str(ticks),'3'
    (directory/'stat').write_text(f'{pid} (owned worker)) '+' '.join(fields))
    for tid,values in {pid:children,**(threads or {})}.items():
        task=directory/'task'/str(tid);task.mkdir(parents=True,exist_ok=True)
        (task/'children').write_text(' '.join(map(str,values)))
    return {'pid':pid,'start_ticks':str(ticks)}


def fixture(tmp_path,monkeypatch,*,finished=False,registered=True,suite_exists=True):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(sys.modules['run_blueprint_followup'],'ROOT',tmp_path)
    proc=tmp_path/'proc';follower=process(proc,100,11,children=[])
    child=process(proc,200,22,ppid=100,children=[300]);task=process(proc,300,33,ppid=200,threads={301:[400]})
    process(proc,400,44,ppid=300)
    source=tmp_path/'source.py';source.write_text('# frozen')
    registration=tmp_path/'runtime/deployment/blueprint_followup_voriconazole_20260920.json'
    suite=tmp_path/pipeline.SUITE_REL
    record={**follower,'output':pipeline.SUITE_REL,'status':'waiting_for_previous_suite',
        'command':[str(tmp_path/'runtime/app-env/bin/python'),str(tmp_path/pipeline.STAGE_REL/'scripts/run_blueprint_reviewed_suite.py'),
            '--manifest',str((tmp_path/pipeline.INPUT_REL).parent/'suite_manifest.json')],
        'source_sha256':{'source.py':pipeline.checksum(source)}}
    status={**record,'status':'completed' if finished else 'running','child':child}
    if registered:
        save(registration,record);save(registration.with_name(registration.stem+'_status.json'),status)
    if suite_exists:
        save(suite/'configuration.json',{'version':'blueprint-reviewed-development-suite-v1','tasks':[{'task_id':'main__toy'}]})
        save(suite/'runner.json',child)
        save(suite/'queue.json',{'tasks':[{'task_id':'main__toy','status':'development_completed' if finished else 'running',
            **task,'output':str((suite/'tasks/main__toy').relative_to(tmp_path))}]})
        save(suite/'progress.json',{'status':'completed' if finished else 'running','stage':'finished' if finished else 'development',
            'active_tasks':[] if finished else ['main__toy']})
        if finished:save(suite/'completion.json',{'status':'finished','task_status_counts':{'development_completed':1}})
    scope=pipeline.ProcessScope(proc_root=proc)
    waiter=pipeline.AwaitMethodsFollower(registration,suite,scope)
    return waiter,proc,record,status


def test_absent_registration_or_not_yet_created_suite_is_bounded_wait_state(tmp_path,monkeypatch):
    waiter,_,record,status=fixture(tmp_path,monkeypatch,registered=False,suite_exists=False)
    assert waiter.sample() is False
    save(waiter.path,record);status.pop('child');status['status']='waiting_for_previous_suite';save(waiter.status_path,status)
    assert waiter.sample() is False and waiter.last_status=='waiting_for_previous_suite'


def test_normal_running_methods_are_read_only_and_never_signaled(tmp_path,monkeypatch):
    waiter,_,_,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(waiter.scope,'send',lambda *_:pytest.fail('Method processes must never be signaled'))
    assert waiter.sample() is False
    assert set(waiter.suite.known)=={200,300,400}


def test_finished_follower_may_be_gone_if_registered_suite_is_fully_released(tmp_path,monkeypatch):
    waiter,proc,_,_=fixture(tmp_path,monkeypatch,finished=True)
    for pid,ticks in ((100,11),(200,22),(300,33),(400,44)):process(proc,pid,ticks,state='Z')
    assert waiter.sample() is True


def test_finished_suite_orphan_prevents_early_prediction_launch(tmp_path,monkeypatch):
    waiter,proc,_,_=fixture(tmp_path,monkeypatch,finished=True)
    assert waiter.sample() is False
    for pid,ticks in ((100,11),(200,22),(300,33)):process(proc,pid,ticks,state='Z')
    process(proc,400,44,ppid=1)
    assert waiter.sample() is False
    process(proc,400,44,state='Z');assert waiter.sample() is True


@pytest.mark.parametrize('damage',['dead','identity','source','suite_identity','stopped'])
def test_dead_or_changed_follower_cannot_bind_other_work(tmp_path,monkeypatch,damage):
    waiter,proc,_,status=fixture(tmp_path,monkeypatch)
    if damage=='dead':process(proc,100,11,state='Z')
    elif damage=='identity':status['start_ticks']='999';save(waiter.status_path,status)
    elif damage=='source':(tmp_path/'source.py').write_text('# changed')
    elif damage=='suite_identity':save(waiter.suite_path/'runner.json',{'pid':999,'start_ticks':'99'})
    else:status['status']='stopped';save(waiter.status_path,status)
    with pytest.raises((ValueError,RuntimeError)):waiter.sample()


def test_follower_completion_race_reloads_status_once(tmp_path,monkeypatch):
    waiter,proc,_,status=fixture(tmp_path,monkeypatch,finished=True)
    for pid,ticks in ((100,11),(200,22),(300,33),(400,44)):process(proc,pid,ticks,state='Z')
    completed=dict(status);status['status']='running';save(waiter.status_path,status)
    real_state=waiter.scope.state
    def state(pid,ticks=None):
        if pid==100:save(waiter.status_path,completed)
        return real_state(pid,ticks)
    monkeypatch.setattr(waiter.scope,'state',state)
    assert waiter.sample() is True


def child_fixture(tmp_path,monkeypatch,*,returncode=0,orphan=False,change_source=False):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    source=tmp_path/'source.py';source.write_text('# frozen');frozen={'source.py':pipeline.checksum(source)}
    sent=[];calls=[];child_identity={'pid':200,'start_ticks':'22','ppid':100};orphan_identity={'pid':300,'start_ticks':'33','ppid':1}
    class Child:
        def __init__(self,command,**_):self.pid=200;self.returncode=None;calls.append(command)
        def poll(self):
            self.returncode=returncode
            if change_source:source.write_text('# changed')
            return self.returncode
        def wait(self,**_):return returncode
    children=[]
    def launch(*a,**kw):
        child=Child(*a,**kw);children.append(child);return child
    class Scope:
        def state(self,pid):return child_identity
        def discover(self,known):
            values=list(known)
            if children and children[0].returncode is None:return {p['pid']:p for p in values}
            return {300:orphan_identity} if orphan else {}
        def send(self,identity,sig):sent.append((identity['pid'],sig));return True
    scope=Scope();actual_stop=pipeline.stop_owned
    monkeypatch.setattr(pipeline.subprocess,'Popen',launch)
    monkeypatch.setattr(pipeline,'stop_owned',lambda s,k:actual_stop(s,k,sleep=lambda _:None))
    kwargs={'name':'predictions','command':['project-python','project-predictor'],'output':tmp_path/'predictions',
        'timeout':30,'byte_limit':1024,'scope':scope,'owner':{'pid':100},'state':{'steps':[]},
        'status_path':tmp_path/'status.json','frozen':frozen,'stopping':lambda:False,'log_path':tmp_path/'prediction.log'}
    return kwargs,sent,calls


def test_child_completion_requires_zero_exit_and_no_live_owned_descendants(tmp_path,monkeypatch):
    kwargs,sent,calls=child_fixture(tmp_path,monkeypatch)
    pipeline.run_child_step(**kwargs)
    assert kwargs['state']['steps'][0]['status']=='completed' and not sent and len(calls)==1


@pytest.mark.parametrize('mode',['failure','orphan','timeout','stop'])
def test_child_failures_stop_only_pipeline_tree_without_retry(tmp_path,monkeypatch,mode):
    kwargs,sent,calls=child_fixture(tmp_path,monkeypatch,returncode=1 if mode=='failure' else 0,orphan=mode=='orphan')
    if mode=='timeout':kwargs['timeout']=-1
    if mode=='stop':kwargs['stopping']=lambda:True
    with pytest.raises(RuntimeError):pipeline.run_child_step(**kwargs)
    assert kwargs['state']['steps'][0]['status']=='failed' and len(calls)==1
    assert {pid for pid,_ in sent}<={200,300}
    assert 100 not in {pid for pid,_ in sent}


def test_source_change_prevents_new_child_launch(tmp_path,monkeypatch):
    kwargs,_,calls=child_fixture(tmp_path,monkeypatch)
    (tmp_path/'source.py').write_text('# changed before prediction')
    with pytest.raises(RuntimeError,match='resource changed'):pipeline.run_child_step(**kwargs)
    assert not calls


def test_source_change_during_last_child_poll_cannot_be_accepted(tmp_path,monkeypatch):
    kwargs,_,calls=child_fixture(tmp_path,monkeypatch,change_source=True)
    with pytest.raises(RuntimeError,match='resource changed'):pipeline.run_child_step(**kwargs)
    assert len(calls)==1
