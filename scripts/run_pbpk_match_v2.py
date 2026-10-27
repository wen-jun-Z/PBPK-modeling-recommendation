#!/usr/bin/env python3
"""Project-owned PBPK-Match v2 worker, resumable execution and terminal progress."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
RUNTIME = ROOT / 'runtime/deployment/pbpk_match_v2'
OUT = ROOT / 'experiments/model_recommendation/pbpk_match_v2'
PYTHON = ROOT / 'runtime/app-env/bin/python'
SCRIPT = Path(__file__).resolve()


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def identity(pid):
    """Inspect ONLY the recorded project worker, never enumerate processes."""
    directory = Path('/proc') / str(int(pid))
    try:
        if directory.stat().st_uid != os.getuid():
            return None
        stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (directory / 'cmdline').read_bytes().split(b'\0')
        cwd = (directory / 'cwd').resolve()
        scripts = [Path(os.fsdecode(arg)) for arg in argv if arg.endswith(b'run_pbpk_match_v2.py')]
        belongs = any((p if p.is_absolute() else cwd / p).resolve() == SCRIPT for p in scripts)
        if stat[0] == 'Z' or not belongs or b'run' not in argv:
            return None
        return {'pid': int(pid), 'start_ticks': stat[19], 'uid': os.getuid()}
    except (OSError, ValueError, IndexError):
        return None


def active():
    recorded = read(RUNTIME / 'worker.json')
    return recorded if recorded and identity(recorded['pid']) == recorded else None


class ProjectLog:
    encoding = 'utf-8'

    def __init__(self):
        self.logger = logging.getLogger('pbpk_match_v2_worker')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = RotatingFileHandler(RUNTIME / 'worker.log', maxBytes=10 * 1024**2, backupCount=2, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def write(self, value):
        if value.strip():
            self.logger.info(value.rstrip())
        return len(value)

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def isatty(self):
        return False


def prepare():
    from pbpk_kb.recommendation.pbpk_match_v2.data import build_dataset, load_dataset
    return load_dataset() if (OUT / 'manifest.json').exists() else build_dataset()


def run(args):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another PBPK-Match v2 worker already holds the project lock')
        write(RUNTIME / 'worker.json', identity(os.getpid()))
        if args.background_log:
            sys.stdout = sys.stderr = ProjectLog()
        from pbpk_kb.recommendation.pbpk_match_v2.experiment import (
            status, freeze_protocol, external_experiment, train_experiment)
        def interrupted(_signum, _frame):
            raise InterruptedError('Project worker received a stop signal')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            if shutil.disk_usage(ROOT).free < 30 * 1024**3:
                raise OSError('Less than 30 GiB free; stop before creating experiment artifacts')
            status('validating', requested_stage=args.stage, batch_size=args.batch_size)
            data = prepare()
            freeze_protocol(data)
            if args.stage in ('all', 'external'):
                external_experiment(data, device=args.device, batch_size=args.batch_size)
            if args.stage in ('all', 'train'):
                train_experiment(data, device=args.device, batch_size=args.batch_size)
            if args.stage in ('all', 'summarize'):
                status('reporting')
                from pbpk_kb.recommendation.pbpk_match_v2.reporting import summarize
                summarize(data, require_complete=True)
            if args.stage == 'all':
                status('publishing')
                from pbpk_kb.recommendation.pbpk_match_v2.publication import publish
                publish()
            status('complete' if args.stage == 'all' else 'stage_complete', completed_stage=args.stage,
                   finished_at=datetime.now(timezone.utc).isoformat())
        except InterruptedError as error:
            # Do not call status while its stop marker is still set.
            previous = read(RUNTIME / 'status.json', {})
            write(RUNTIME / 'status.json', {**previous, 'phase': 'stopped', 'reason': str(error),
                  'updated_at': datetime.now(timezone.utc).isoformat()})
            print(str(error), flush=True)
        except BaseException as error:
            previous = read(RUNTIME / 'status.json', {})
            write(RUNTIME / 'status.json', {**previous, 'phase': 'failed', 'error_type': type(error).__name__,
                  'error': str(error), 'updated_at': datetime.now(timezone.utc).isoformat()})
            traceback.print_exc()
            raise


def start(args):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if active():
        print('PBPK-Match v2 已运行：PID', active()['pid'])
        return
    (RUNTIME / 'stop.requested').unlink(missing_ok=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                       HF_HUB_DISABLE_IMPLICIT_TOKEN='1', MPLCONFIGDIR=str(ROOT / 'runtime/cache/pbpk_match_v2/matplotlib'))
    command = [str(PYTHON), '-B', str(SCRIPT), 'run', '--stage', args.stage,
               '--device', args.device, '--batch-size', str(args.batch_size), '--background-log']
    with (RUNTIME / 'launcher.log').open('ab') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True)
    time.sleep(.5)
    if child.poll() is not None:
        raise SystemExit('Worker exited at launch; inspect project launcher.log/worker.log')
    print(f'PBPK-Match v2 已启动，PID {child.pid}；SSH 断开后继续。')
    print('进度：runtime/app-env/bin/python -B scripts/run_pbpk_match_v2.py watch')


def display():
    state = read(RUNTIME / 'status.json', {})
    jobs = len(list((OUT / 'runs').glob('*/*/*/*/complete.json')))
    external = len(list((OUT / 'external').glob('*/complete.json')))
    worker = active()
    def bar(done, total):
        fraction = min(done / max(total, 1), 1.)
        filled = round(32 * fraction)
        return f'┃{"█" * filled}{"░" * (32 - filled)}┃ {done:>3}/{total:<3} {fraction:6.1%}'
    rows = ['PBPK-Match v2  ·  内部分组验证',
            f'进程：{"运行中 PID " + str(worker["pid"]) if worker else "未运行"}    阶段：{state.get("phase", "未启动")}',
            f'冻结对照  {bar(external, 9)}', f'分组拟合  {bar(jobs, 300)}',
            f'当前：{state.get("method", "—")}  seed={state.get("seed", "—")}  fold={state.get("fold", "—")}']
    if state.get('phase') == 'external_inference':
        total = state.get('completed_rows', 0) + state.get('remaining_rows', 0)
        rows.append(f'推理续跑  {bar(state.get("completed_rows", 0), total)}  batch={state.get("batch_size")}')
    progress = state.get('training_progress', {})
    if 'training' in state.get('phase', ''):
        if 'completed_queries' in progress:
            rows.append('查询进度  ' + bar(progress['completed_queries'], progress['total_queries']) + '  ' + progress['stage'])
        elif 'pairs_this_epoch' in progress:
            rows.append('本轮样本  ' + bar(progress['pairs_this_epoch'], state.get('training_pairs_per_epoch', 0)))
        rows.append(f'epoch {progress.get("epoch", "—")}/{state.get("maximum_epochs", "—")}  '
                    f'已用 {progress.get("elapsed_seconds", 0) / 60:.1f} min')
    if state.get('error'):
        rows.append('失败原因：' + state['error'])
    rows += [f'更新时间：{state.get("updated_at", "—")}', '日志：runtime/deployment/pbpk_match_v2/worker.log',
             'Ctrl+C 只关闭进度显示，不停止实验。']
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'start', 'stop', 'status', 'watch', 'summarize'])
    parser.add_argument('--stage', choices=['all', 'external', 'train', 'summarize'], default='all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--background-log', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Batch size must be positive')
    if args.action in ('run', 'start') and args.stage in ('train', 'all') and args.batch_size < 2:
        parser.error('Training requires batch size >= 2')
    if args.action == 'prepare':
        data = prepare()
        print(f'数据验证通过：{len(data["query_records"])} 请求 × {len(data["candidate_records"])} 候选')
    elif args.action == 'start':
        start(args)
    elif args.action == 'run':
        run(args)
    elif args.action == 'stop':
        if active():
            (RUNTIME / 'stop.requested').write_text('User requested project stop\n')
            print('已请求在下一进度检查点停止；已有断点保留。')
        else:
            print('PBPK-Match v2 未运行。')
    elif args.action == 'status':
        print(display())
    elif args.action == 'summarize':
        args.stage = 'summarize'
        run(args)
    elif args.action == 'watch':
        try:
            while True:
                print(('\033[2J\033[H' if sys.stdout.isatty() else '') + display(), flush=True)
                time.sleep(2)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
