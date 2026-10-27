"""Project-tree watchdog tests with fake procfs; no suite process is signaled."""
from datetime import UTC, datetime, timedelta
import importlib.util
import json
import os
from pathlib import Path
import signal

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / 'scripts/watch_blueprint_suite.py'
spec = importlib.util.spec_from_file_location('suite_watchdog_under_test', MODULE_PATH)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def process(proc, pid, ticks, *, ppid=1, state='S', children=(), threaded_children=None, cpu=10, rss=3):
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ['0'] * 22
    fields[0], fields[1], fields[11], fields[12] = state, str(ppid), str(cpu), '0'
    fields[19], fields[21] = str(ticks), str(rss)
    (directory/'stat').write_text(f'{pid} (owned worker)) ' + ' '.join(fields))
    for tid, values in {pid: children, **(threaded_children or {})}.items():
        task = directory/'task'/str(tid)
        task.mkdir(parents=True, exist_ok=True)
        (task/'children').write_text(' '.join(map(str, values)))
    return {'pid': pid, 'start_ticks': str(ticks)}


def test_discovers_children_created_by_nonleader_threads_recursively(tmp_path):
    proc = tmp_path/'proc'
    root = process(proc, 100, 11, children=[200])
    process(proc, 200, 22, ppid=100, children=[], threaded_children={201: [300]})
    process(proc, 300, 33, ppid=200, children=[400])
    process(proc, 400, 44, ppid=300)
    process(proc, 900, 99)  # Never referenced by a known task.
    result = watch.ProcessScope(proc_root=proc).discover([root])
    assert set(result) == {100, 200, 300, 400}


def test_retains_previously_authenticated_orphan_but_rejects_pid_reuse(tmp_path):
    proc = tmp_path/'proc'
    root = process(proc, 100, 11, children=[200])
    child = process(proc, 200, 22, ppid=100)
    scope = watch.ProcessScope(proc_root=proc)
    tracked = scope.discover([root])
    process(proc, 100, 999)  # Same PID now names an unrelated process.
    process(proc, 200, 22, ppid=1, children=[300])
    process(proc, 300, 33, ppid=200)
    assert set(scope.discover(tracked.values())) == {200, 300}
    assert scope.state(child['pid'], 'different') is None


def test_ownership_and_parent_mismatch_never_adopt_an_unrelated_process(tmp_path):
    proc = tmp_path/'proc'
    root = process(proc, 100, 11, children=[200])
    process(proc, 200, 22, ppid=999)
    assert set(watch.ProcessScope(proc_root=proc).discover([root])) == {100}
    assert watch.ProcessScope(proc_root=proc, uid=os.getuid()+1).state(100) is None


def test_zombies_are_not_alive_and_missing_tasks_are_tolerated(tmp_path):
    proc = tmp_path/'proc'
    root = process(proc, 100, 11, children=[200, 300])
    process(proc, 200, 22, ppid=100, state='Z')
    scope = watch.ProcessScope(proc_root=proc)
    assert set(scope.discover([root])) == {100}
    (proc/'100/task/100/children').unlink()
    assert scope.children(root) == []


@pytest.mark.parametrize('reused', [False, True])
def test_pidfd_signal_rechecks_identity_after_open_and_never_uses_kill(tmp_path, monkeypatch, reused):
    scope = watch.ProcessScope(proc_root=tmp_path)
    identity = {'pid': 100, 'start_ticks': '11'}
    states = iter([identity, None if reused else identity])
    monkeypatch.setattr(scope, 'state', lambda *args: next(states))
    sent, closed = [], []
    monkeypatch.setattr(watch.os, 'pidfd_open', lambda pid, flags: 123456)
    monkeypatch.setattr(watch.signal, 'pidfd_send_signal', lambda fd, sig, info, flags: sent.append((fd, sig)))
    monkeypatch.setattr(watch.os, 'close', lambda fd: closed.append(fd))
    monkeypatch.setattr(watch.os, 'kill', lambda *args: pytest.fail('PID-based kill is forbidden'))
    assert scope.send(identity, signal.SIGTERM) is (not reused)
    assert sent == ([] if reused else [(123456, signal.SIGTERM)])
    assert closed == [123456]


def test_scoped_resource_samples_include_worker_cpu_without_counting_pid_reuse():
    before = {1: {'pid': 1, 'start_ticks': '10', 'cpu_ticks': 100, 'rss_pages': 2},
              2: {'pid': 2, 'start_ticks': '20', 'cpu_ticks': 200, 'rss_pages': 3}}
    after = {1: dict(before[1], cpu_ticks=150), 2: dict(before[2], cpu_ticks=400),
             3: {'pid': 3, 'start_ticks': '30', 'cpu_ticks': 800, 'rss_pages': 4}}
    sample = watch.resource_sample(after, before, 1, hz=100, page_size=4096)
    assert sample['owned_process_count'] == 3
    assert sample['sampled_cpu_percent_of_one_core'] == 250
    assert sample['summed_process_rss_bytes'] == 9*4096
    after[2]['start_ticks'] = 'reused'
    assert watch.resource_sample(after, before, 1, hz=100, page_size=4096)['sampled_cpu_percent_of_one_core'] == 50


def fixture_watchdog(tmp_path):
    proc = tmp_path/'proc'
    identity = process(proc, 100, 11, children=[200])
    process(proc, 200, 22, ppid=100, threaded_children={201: [300]})
    process(proc, 300, 33, ppid=200)
    output = tmp_path/'experiments/suite'
    output.mkdir(parents=True)
    deployment = tmp_path/'runtime/deployment'
    deployment.mkdir(parents=True)
    timestamp = datetime.now(UTC).isoformat()
    registration = {**identity, 'started_at': timestamp, 'output': 'experiments/suite',
                    'log': 'runtime/deployment/suite.log'}
    config = {'version': 'blueprint-reviewed-development-suite-v1', 'max_output_bytes': 1024**2,
              'max_task_seconds': 3600, 'tasks': [{'task_id': 'main__toy'}]}
    queue = {'tasks': [{'task_id': 'main__toy', 'pid': 200, 'start_ticks': '22', 'status': 'running',
                       'started_at': timestamp, 'output': 'experiments/suite/tasks/main__toy',
                       'log': 'experiments/suite/tasks/main__toy.log'}]}
    for path, value in ((deployment/'registration.json', registration), (output/'runner.json', identity),
                        (output/'configuration.json', config), (output/'queue.json', queue)):
        path.write_text(json.dumps(value))
    watcher = watch.SuiteWatchdog(deployment/'registration.json', project_root=tmp_path,
                                  scope=watch.ProcessScope(proc_root=proc))
    return watcher, proc, output


def test_healthy_suite_is_observed_without_any_signal(tmp_path, monkeypatch):
    watcher, _, _ = fixture_watchdog(tmp_path)
    monkeypatch.setattr(watcher.scope, 'send', lambda *args: pytest.fail('Do not stop a healthy suite'))
    snapshot = watcher.sample()
    assert snapshot['status'] == 'running'
    assert snapshot['reason'] is None
    assert snapshot['resources']['owned_process_count'] == 3


def test_supervisor_loss_keeps_known_orphans_and_cleanup_is_scoped(tmp_path, monkeypatch):
    watcher, proc, _ = fixture_watchdog(tmp_path)
    watcher.sample()
    process(proc, 100, 999, children=[900])
    process(proc, 900, 99, ppid=100)  # Child of reused, unrelated supervisor PID.
    process(proc, 200, 22, ppid=1, threaded_children={201: [300]})
    snapshot = watcher.sample()
    assert snapshot['reason'] == 'supervisor_exited_without_completion'
    sent = []
    monkeypatch.setattr(watcher.scope, 'send', lambda identity, sig: sent.append((identity['pid'], sig)) or True)
    watcher.terminate_known(grace_seconds=0)
    assert {pid for pid, _ in sent} == {200, 300}
    assert {sig for _, sig in sent} == {signal.SIGTERM, signal.SIGKILL}


def test_finished_supervisor_is_distinguished_from_unexpected_exit(tmp_path):
    watcher, proc, output = fixture_watchdog(tmp_path)
    (output/'completion.json').write_text(json.dumps({'status': 'finished'}))
    process(proc, 100, 11, state='Z')
    assert watcher.sample()['reason'] == 'supervisor_finished'


def test_output_and_wall_time_limits_are_frozen_inputs(tmp_path):
    watcher, _, output = fixture_watchdog(tmp_path)
    (output/'bounded_fixture').write_bytes(b'x' * (1024**2))
    assert watcher.sample()['reason'] == 'suite_output_byte_limit'
    (output/'bounded_fixture').unlink()
    watcher.registration['started_at'] = (datetime.now(UTC)-timedelta(hours=25)).isoformat()
    assert watcher.sample()['reason'] == 'suite_24_hour_limit'


def test_registration_mismatch_and_outside_paths_are_rejected(tmp_path):
    watcher, _, output = fixture_watchdog(tmp_path)
    (output/'runner.json').write_text(json.dumps({'pid': 100, 'start_ticks': 'different'}))
    with pytest.raises(ValueError, match='identities differ'):
        watch.SuiteWatchdog(watcher.registration_path, project_root=tmp_path)
    with pytest.raises(ValueError, match='remain in this project'):
        watch.inside('../outside', root=tmp_path)


def test_monitor_output_is_atomic_bounded_and_logs_rotate(tmp_path, monkeypatch):
    monkeypatch.setattr(watch, 'MAX_MONITOR_LOG_BYTES', 80)
    path = tmp_path/'monitor.json'
    watch.write_json(path, {'status': 'running'})
    watch.write_json(path, {'status': 'completed'})
    assert json.loads(path.read_text()) == {'status': 'completed'}
    assert not list(tmp_path.glob('.*.tmp'))
    log = tmp_path/'monitor.log'
    for index in range(20):
        watch.append_event(log, {'event': 'bounded', 'index': index})
    assert log.stat().st_size <= 80
    assert log.with_name('monitor.log.1').stat().st_size <= 80


def test_output_scan_never_follows_symlink(tmp_path):
    owned, other = tmp_path/'owned', tmp_path/'other'
    owned.mkdir(); other.mkdir()
    (owned/'link').symlink_to(other, target_is_directory=True)
    with pytest.raises(RuntimeError, match='unsafe symlink'):
        watch.directory_bytes(owned)
