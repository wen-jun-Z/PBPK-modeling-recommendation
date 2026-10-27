"""Bounded, project-only watchdog for an already registered blueprint suite.

Linux children belong to their creating task/TID: inspecting only the leader's
children omits R workers spawned by Python thread pools. This watcher follows
all tasks of verified project processes, never enumerates /proc processes, and
never reads command lines or environments. Previously observed descendants keep
their identity after reparenting. Unobserved processes that both detach and lose
all observed ancestors between polls cannot be discovered without a cgroup;
this watcher does not claim that stronger guarantee.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import signal
import stat
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = 'runtime/deployment/blueprint_reviewed_suite_20260920.json'
MAX_PROCESSES = 4096
MAX_JSON_BYTES = 2 * 1024**2
MAX_MONITOR_LOG_BYTES = 512 * 1024


def inside(value, *, root=ROOT):
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Watchdog paths must remain in this project')
    return path


def read_json(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > MAX_JSON_BYTES:
        raise ValueError('Watchdog JSON must be a bounded, owned regular file')
    return json.loads(path.read_text())


def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if len(data) > MAX_JSON_BYTES:
        raise ValueError('Monitor snapshot exceeded its byte cap')
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def append_event(path, event):
    """One bounded active log and one bounded previous log, never unbounded append."""
    data = (json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if len(data) > 4096:
        raise ValueError('Monitor event exceeded its line cap')
    if path.exists():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError('Unsafe watchdog log')
        if info.st_size + len(data) > MAX_MONITOR_LOG_BYTES:
            previous = path.with_name(path.name + '.1')
            if previous.is_symlink():
                raise ValueError('Unsafe watchdog rotated log')
            path.replace(previous)
    with path.open('ab') as stream:
        stream.write(data)


class ProcessScope:
    def __init__(self, *, proc_root=Path('/proc'), uid=None):
        self.proc_root = Path(proc_root)
        self.uid = os.getuid() if uid is None else uid

    def state(self, pid, expected_ticks=None):
        if type(pid) is not int or pid <= 0:
            return None
        directory = self.proc_root / str(pid)
        try:
            if directory.stat().st_uid != self.uid:
                return None
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            value = {'pid': pid, 'start_ticks': fields[19], 'ppid': int(fields[1]),
                     'state': fields[0], 'cpu_ticks': int(fields[11]) + int(fields[12]),
                     'rss_pages': max(0, int(fields[21]))}
            if expected_ticks is not None and value['start_ticks'] != str(expected_ticks):
                return None
            if value['state'] in {'Z', 'X', 'x'}:
                return None
            return value
        except (FileNotFoundError, ProcessLookupError):
            return None

    def children(self, identity):
        """Inspect only task directories of one still-matching owned process."""
        pid, ticks = identity['pid'], identity['start_ticks']
        if self.state(pid, ticks) is None:
            return []
        children = set()
        try:
            tasks = self.proc_root / str(pid) / 'task'
            for task in tasks.iterdir():
                if not task.name.isdecimal():
                    continue
                try:
                    children.update(int(p) for p in (task / 'children').read_text().split())
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if len(children) > MAX_PROCESSES:
                    raise RuntimeError('Project descendant limit exceeded')
        except (FileNotFoundError, ProcessLookupError):
            return []
        # The parent may exit while task files are being read. Do not adopt
        # children discovered from a reused process directory.
        if self.state(pid, ticks) is None:
            return []
        result = []
        for child in sorted(children):
            value = self.state(child)
            if value is not None and value['ppid'] == pid:
                result.append(value)
        return result if self.state(pid, ticks) is not None else []

    def discover(self, identities):
        """Seeds must already have authenticated project ancestry/registration."""
        pending, found = list(identities), {}
        while pending:
            identity = pending.pop()
            current = self.state(identity['pid'], identity['start_ticks'])
            if current is None or current['pid'] in found:
                continue
            found[current['pid']] = current
            if len(found) > MAX_PROCESSES:
                raise RuntimeError('Project descendant limit exceeded')
            pending.extend(self.children(current))
        return found

    def send(self, identity, sig):
        """pidfd binds the signal to one process instance, closing PID reuse races."""
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise RuntimeError('Safe termination requires Linux pidfd support')
        if self.state(identity['pid'], identity['start_ticks']) is None:
            return False
        descriptor = None
        try:
            descriptor = os.pidfd_open(identity['pid'], 0)
            if self.state(identity['pid'], identity['start_ticks']) is None:
                return False
            signal.pidfd_send_signal(descriptor, sig, None, 0)
            return True
        except ProcessLookupError:
            return False
        finally:
            if descriptor is not None:
                os.close(descriptor)


def directory_bytes(directory):
    """No symlink following; inspect only the explicitly registered output tree."""
    total, entries, pending = 0, 0, [directory]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as children:
                for entry in children:
                    entries += 1
                    if entries > 500000:
                        raise RuntimeError('Suite output entry cap exceeded')
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if stat.S_ISLNK(info.st_mode):
                        raise RuntimeError('Suite output contains an unsafe symlink')
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(info.st_mode):
                        total += info.st_size
        except FileNotFoundError:
            continue
    return total


def seconds_since(value, now):
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ValueError('Registered timestamps must have a time zone')
    return max(0., (now - timestamp).total_seconds())


def resource_sample(current, previous, elapsed, *, hz, page_size):
    delta = sum(max(0, value['cpu_ticks'] - previous[pid]['cpu_ticks'])
                for pid, value in current.items() if pid in previous
                and previous[pid]['start_ticks'] == value['start_ticks'])
    return {'owned_process_count': len(current),
            'sampled_cpu_percent_of_one_core': 100 * delta / hz / elapsed if previous and elapsed > 0 else None,
            'summed_process_rss_bytes': sum(v['rss_pages'] for v in current.values()) * page_size,
            'measurement_scope': 'verified project processes; all task/TID children; no global process enumeration',
            'limitations': 'CPU omits processes that exit between samples and the first interval of newly discovered processes; RSS sums shared pages'}


class SuiteWatchdog:
    def __init__(self, registration, *, project_root=ROOT, scope=None):
        self.root = Path(project_root).resolve()
        self.registration_path = inside(registration, root=self.root)
        self.registration = read_json(self.registration_path)
        self.supervisor = {k: self.registration[k] for k in ('pid', 'start_ticks')}
        if type(self.supervisor['pid']) is not int or not str(self.supervisor['start_ticks']).isdecimal():
            raise ValueError('Explicit supervisor PID/start ticks are required')
        self.output = inside(self.registration['output'], root=self.root)
        self.log = inside(self.registration['log'], root=self.root)
        self.config = read_json(self.output / 'configuration.json')
        own_registration = read_json(self.output / 'runner.json')
        if any(str(own_registration[k]) != str(self.supervisor[k]) for k in ('pid', 'start_ticks')):
            raise ValueError('Deployment and experiment supervisor identities differ')
        if self.config.get('version') != 'blueprint-reviewed-development-suite-v1':
            raise ValueError('Watch only a registered reviewed development suite')
        if not 1 <= self.config['max_output_bytes'] <= 32 * 1024**3 or not 360 <= self.config['max_task_seconds'] <= 8*3600:
            raise ValueError('Unsafe frozen suite resource limits')
        self.scope = scope or ProcessScope()
        self.known, self.previous = {}, {}
        self.last_sample = time.monotonic()
        prior = self.output / 'monitor.json'
        if prior.is_file():
            previous = read_json(prior)
            if previous.get('supervisor') == self.supervisor:
                tracked = previous.get('tracked_processes', [])
                if len(tracked) > MAX_PROCESSES:
                    raise ValueError('Prior monitor identity count exceeds limit')
                self.known = {p['pid']: p for p in tracked}
        self.events = self.output / 'monitor.log'

    def registered_task_roots(self):
        queue_path = self.output / 'queue.json'
        if not queue_path.is_file():
            return [], []
        tasks = read_json(queue_path)['tasks']
        allowed = {t['task_id'] for t in self.config['tasks']}
        roots = []
        for task in tasks:
            if task['task_id'] not in allowed:
                raise ValueError('Queue contains a task outside the frozen suite')
            if task.get('pid') and task.get('start_ticks'):
                if task.get('output') and not inside(task['output'], root=self.root).is_relative_to(self.output):
                    raise ValueError('Task output leaves the registered suite')
                roots.append({'pid': task['pid'], 'start_ticks': task['start_ticks']})
        return roots, tasks

    def sample(self):
        now = datetime.now(UTC)
        monotonic = time.monotonic()
        roots, tasks = self.registered_task_roots()
        self.known = self.scope.discover([self.supervisor, *self.known.values(), *roots])
        resources = resource_sample(self.known, self.previous, monotonic-self.last_sample,
                                    hz=os.sysconf('SC_CLK_TCK'), page_size=os.sysconf('SC_PAGE_SIZE'))
        self.previous, self.last_sample = dict(self.known), monotonic
        alive = self.scope.state(self.supervisor['pid'], self.supervisor['start_ticks']) is not None
        reason = None
        total_bytes = directory_bytes(self.output)
        if total_bytes > self.config['max_output_bytes']:
            reason = 'suite_output_byte_limit'
        elif seconds_since(self.registration['started_at'], now) > 24*3600:
            reason = 'suite_24_hour_limit'
        paths = [self.log]
        for task in tasks:
            if task.get('log'):
                path = inside(task['log'], root=self.root)
                if not path.is_relative_to(self.output):
                    raise ValueError('Task log leaves the registered suite')
                paths.append(path)
            if (task.get('status') == 'running' and task.get('started_at')
                    and seconds_since(task['started_at'], now) > self.config['max_task_seconds']):
                reason = reason or 'task_wall_time_limit:' + task['task_id']
        for path in paths:
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('Registered suite log must be a regular file')
                if info.st_size > 20*1024**2:
                    reason = reason or 'registered_log_byte_limit'
            except FileNotFoundError:
                pass
        completion_path = self.output / 'completion.json'
        completion = read_json(completion_path).get('status') if completion_path.is_file() else None
        if not alive:
            reason = reason or ('supervisor_finished' if completion == 'finished' else 'supervisor_exited_without_completion')
        snapshot = {'version': 1, 'sampled_at': now.isoformat(), 'supervisor': self.supervisor,
                    'supervisor_alive': alive, 'status': 'running' if reason is None else 'action_required',
                    'reason': reason, 'output_bytes': total_bytes, 'output_limit_bytes': self.config['max_output_bytes'],
                    'resources': resources, 'tracked_processes': sorted(self.known.values(), key=lambda v: v['pid']),
                    'polling_seconds': 10, 'scientific_results_modified': False}
        return snapshot

    def terminate_known(self, *, grace_seconds=3):
        """Stop only authenticated project processes, including tracked orphans."""
        targets = self.scope.discover([self.supervisor, *self.known.values()])
        sent = {'term': 0, 'kill': 0}
        for label, sig in (('term', signal.SIGTERM), ('kill', signal.SIGKILL)):
            ordered = sorted(targets.values(), key=lambda p: p['pid'] == self.supervisor['pid'])
            for identity in ordered:
                sent[label] += self.scope.send(identity, sig)
            if label == 'term':
                deadline = time.monotonic() + grace_seconds
                while time.monotonic() < deadline:
                    targets = self.scope.discover(list(targets.values()))
                    if not targets:
                        break
                    time.sleep(min(.2, max(0., deadline-time.monotonic())))
            targets = self.scope.discover(list(targets.values()))
        # pidfd_send_signal returns before process teardown. Observe termination
        # briefly instead of claiming cleanup while a verified worker is alive.
        deadline = time.monotonic() + min(grace_seconds, 1.)
        while targets and time.monotonic() < deadline:
            time.sleep(min(.1, max(0., deadline-time.monotonic())))
            targets = self.scope.discover(list(targets.values()))
        self.known = targets
        return {'signals_sent': sent, 'remaining_verified_processes': sorted(targets.values(), key=lambda p: p['pid'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=Path(REGISTRATION))
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--once', action='store_true', help='Read and publish one sample without signaling any process')
    args = parser.parse_args(argv)
    if not 1 <= args.interval <= 20:
        raise ValueError('Polling interval must lie in 1..20 seconds')
    watcher = SuiteWatchdog(args.registration)
    stopping = False
    def stop_monitor(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop_monitor); signal.signal(signal.SIGINT, stop_monitor)
    started, previous_status = time.monotonic(), None
    append_event(watcher.events, {'at': datetime.now(UTC).isoformat(), 'event': 'monitor_started', 'once': args.once})
    while not stopping:
        tick = time.monotonic()
        try:
            snapshot = watcher.sample()
        except (OSError, ValueError, RuntimeError) as exc:
            snapshot = {'version': 1, 'supervisor': watcher.supervisor, 'status': 'action_required',
                        'reason': 'monitor_check_failed:' + type(exc).__name__,
                        'tracked_processes': list(watcher.known.values())}
        snapshot['polling_seconds'] = args.interval
        if time.monotonic()-started >= 24*3600:
            snapshot.update(status='action_required', reason='monitor_24_hour_limit')
        if snapshot.get('reason') and not args.once:
            try:
                cleanup = watcher.terminate_known()
                status = ('cleanup_incomplete' if cleanup['remaining_verified_processes'] else
                          'completed' if snapshot['reason'] == 'supervisor_finished' else 'stopped')
                snapshot.update(status=status, cleanup=cleanup, tracked_processes=list(watcher.known.values()))
            except (OSError, ValueError, RuntimeError) as exc:
                snapshot.update(status='cleanup_failed', cleanup_error_type=type(exc).__name__,
                                tracked_processes=list(watcher.known.values()))
        write_json(watcher.output / 'monitor.json', snapshot)
        status = (snapshot['status'], snapshot.get('reason'))
        if status != previous_status:
            append_event(watcher.events, {'at': datetime.now(UTC).isoformat(), 'event': 'monitor_state',
                                         'status': status[0], 'reason': status[1]})
            previous_status = status
        if args.once or snapshot.get('reason'):
            return 0 if snapshot['status'] in {'running', 'completed'} else 1
        while not stopping and time.monotonic()-tick < args.interval:
            time.sleep(min(.5, args.interval-(time.monotonic()-tick)))
    append_event(watcher.events, {'at': datetime.now(UTC).isoformat(), 'event': 'monitor_stopped_without_stopping_suite'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
