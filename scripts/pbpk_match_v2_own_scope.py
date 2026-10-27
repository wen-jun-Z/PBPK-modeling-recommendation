#!/usr/bin/env python3
"""Limit the existing frozen worker to PBPK-Match v2 without restarting a fit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / 'src'))
import run_pbpk_match_v2 as worker

RUNTIME, OUT = worker.RUNTIME, worker.OUT
SCOPE = RUNTIME / 'scope.json'
STATUS = RUNTIME / 'scope_status.json'
SEEDS = (17, 23, 41)
DEFERRED = ('no_numeric', 'frozen_text', 'mlp', 'lambdamart')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def monitor_alive():
    ref = worker.read(RUNTIME / 'scope_worker.json')
    if not ref:
        return False
    try:
        p = Path('/proc') / str(ref['pid'])
        if p.stat().st_uid != os.getuid():
            return False
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        args = (p / 'cmdline').read_bytes().split(b'\0')
        return stat[0] != 'Z' and stat[19] == ref['start_ticks'] and str(SCRIPT).encode() in args and b'monitor' in args
    except OSError:
        return False


def check_scope():
    scope = worker.read(SCOPE)
    if not scope or scope['mode'] != 'own_model_only':
        raise ValueError('Own-model execution scope is missing or disabled')
    if scope['original_protocol_sha256'] != sha(OUT / 'protocol.json') or scope['control_script_sha256'] != sha(SCRIPT):
        raise ValueError('Scope protocol or controller changed')
    return scope


def apply():
    protocol = worker.read(OUT / 'protocol.json')
    if protocol['methods'] != ['full', *DEFERRED] or protocol['seeds'] != list(SEEDS):
        raise ValueError('This controller requires the existing full-first protocol')
    for path, expected in protocol['implementation_sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Frozen implementation changed: ' + path)
    if not SCOPE.exists():
        worker.write(SCOPE, {'schema': 'pbpk-match-v2-execution-scope-1', 'mode': 'own_model_only',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'original_protocol_sha256': sha(OUT / 'protocol.json'), 'active_methods': ['full'],
            'deferred_methods': list(DEFERRED), 'seeds': list(SEEDS), 'outer_folds': 5,
            'inner_folds': 3, 'expected_fit_jobs': 60, 'external_methods': protocol['external_methods'],
            'control_script_sha256': sha(SCRIPT), 'frozen_training_and_labels_unchanged': True,
            'completion_scope': 'full three-seed grouped validation plus existing external results; not the original 300 jobs'})
    check_scope()
    if monitor_alive():
        print('仅自建模型范围约束已生效。')
        return
    if not worker.active():
        raise RuntimeError('The existing project training worker is not active; no unrelated process will be started')
    with (RUNTIME / 'scope.log').open('ab') as log:
        child = subprocess.Popen([str(worker.PYTHON), '-B', str(SCRIPT), 'monitor'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    time.sleep(.3)
    if child.poll() is not None:
        raise RuntimeError('Scope monitor could not start; inspect its project log')
    print(f'仅继续 PBPK-Match v2；范围监控 PID {child.pid}。当前训练不重启。')


def decision(state, last_prediction_new, outer_complete, seeds_complete):
    """Stop after the final full prediction, before any deferred method."""
    if state.get('method') in DEFERRED:
        return 'deferred_method_boundary'
    if seeds_complete or outer_complete:
        return 'full_artifacts_finished'
    if (last_prediction_new and state.get('method') == 'full' and state.get('seed') == 41
            and state.get('fold') == 4):
        return 'final_full_prediction_barrier'
    return None


def monitor():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'scope.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stat = (Path('/proc') / str(os.getpid()) / 'stat').read_text().rsplit(')', 1)[1].split()
        worker.write(RUNTIME / 'scope_worker.json', {'pid': os.getpid(), 'start_ticks': stat[19], 'uid': os.getuid()})
        scope = check_scope()
        created = datetime.fromisoformat(scope['created_at_utc']).timestamp()
        final = OUT / 'runs/full/seed41/fold4/outer'
        last_message = None
        stopped_for_scope = False
        try:
            while True:
                state = worker.read(RUNTIME / 'status.json', {})
                refs = [OUT / 'predictions' / f'full_seed{seed}.json' for seed in SEEDS]
                seeds_ready = all(p.exists() for p in refs)
                last_predictions = final / 'predictions.npz'
                new_predictions = last_predictions.exists() and last_predictions.stat().st_mtime >= created
                reason = decision(state, new_predictions, (final / 'complete.json').exists(), seeds_ready)
                active = worker.active()
                if reason and active:
                    marker = RUNTIME / 'stop.requested'
                    if not marker.exists():
                        marker.write_text('Own-model scope complete; deferred methods must not run\n')
                    stopped_for_scope = True
                    # Final prediction -> integrity -> seed merge contains no stop
                    # checks. The existing worker stops at seed_complete after
                    # saving every full result. A deferred method is stopped now.
                    if reason == 'deferred_method_boundary' and worker.identity(active['pid']) == active:
                        os.kill(active['pid'], signal.SIGTERM)
                phase = 'finishing_full_results' if reason else 'full_training_continues'
                message = {'state': phase, 'mode': 'own_model_only', 'training_pid': active['pid'] if active else None,
                           'active_methods': ['full'], 'deferred_methods': list(DEFERRED),
                           'reason': reason, 'expected_fit_jobs': 60}
                if message != last_message:
                    worker.write(STATUS, {**message, 'updated_at_utc': datetime.now(timezone.utc).isoformat()})
                    print(json.dumps(message), flush=True)
                    last_message = message
                if not active:
                    if seeds_ready and (stopped_for_scope or state.get('phase') == 'stopped'):
                        from summarize_pbpk_match_v2_own import summarize
                        summarize(scope_path=SCOPE)
                        worker.write(STATUS, {**message, 'state': 'complete',
                            'report': str(OUT / 'report_own_only'), 'original_300_jobs_complete': False})
                    else:
                        worker.write(STATUS, {**message, 'state': 'worker_stopped_before_own_completion',
                                             'worker_phase': state.get('phase'), 'worker_error': state.get('error')})
                    return
                time.sleep(.5)
        except BaseException as error:
            if worker.active():
                (RUNTIME / 'stop.requested').write_text('Scope monitor failed; pause project worker safely\n')
            worker.write(STATUS, {'state': 'failed', 'error': str(error)})
            raise


def display():
    s = worker.read(RUNTIME / 'status.json', {})
    jobs = len(list((OUT / 'runs/full').glob('*/*/*/complete.json')))
    filled = round(32 * min(jobs, 60) / 60)
    p = s.get('training_progress', {})
    return '\n'.join(['PBPK-Match v2 · 仅自建模型',
        f'自身拟合  ┃{"█"*filled}{"░"*(32-filled)}┃ {jobs}/60  {jobs/60:.1%}',
        f'当前：{s.get("method", "—")}  seed={s.get("seed", "—")}  外层折={s.get("fold", "—")}  epoch={p.get("epoch", "—")}',
        f'阶段：{s.get("phase", "—")}  范围监控：{"运行中" if monitor_alive() else worker.read(STATUS, {}).get("state", "未运行")}',
        '9 个已有对照结果直接复用；其他训练基线和消融暂缓。',
        f'更新时间：{s.get("updated_at", "—")}', 'Ctrl+C 仅关闭进度显示。'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['apply', 'monitor', 'status', 'watch'])
    args = parser.parse_args()
    if args.action == 'apply':
        apply()
    elif args.action == 'monitor':
        monitor()
    elif args.action == 'status':
        print(display())
    else:
        try:
            while True:
                print(('\033[2J\033[H' if sys.stdout.isatty() else '') + display(), flush=True)
                time.sleep(2)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
