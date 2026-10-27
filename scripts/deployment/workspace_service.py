"""Manage only the private PBPK web workspace; never starts GPU experiments."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import secrets
import select
import signal
import socket
import subprocess
import sys
import time

from service_runner import ROOT, DEPLOY, environment

RUN = DEPLOY / 'workspace'
PY = ROOT / 'runtime/app-env/bin/python'
PID = RUN / 'pid.json'


def record():
    try:
        return json.loads(PID.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def alive(row):
    try:
        p = Path('/proc') / str(row['pid'])
        if p.stat().st_uid != os.getuid() or (p/'cwd').resolve() != ROOT:
            return False
        fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == row['start_ticks']
    except (OSError, KeyError):
        return False


def ensure_database(env):
    """Only the PostgreSQL data directory already belonging to this project."""
    import psycopg
    from pgserver._commands import POSTGRES_BIN_PATH
    data = ROOT / 'runtime/postgres-data'
    ctl = str(POSTGRES_BIN_PATH / 'pg_ctl')
    status = subprocess.run([ctl, '-D', str(data), 'status'], env=env, cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if status.returncode != 0:
        ports = json.loads((DEPLOY/'ports.json').read_text())
        port = ports['postgres']
        # Check by binding only. Never inspect any conflicting service.
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                raise RuntimeError('Project database port unavailable; no other listener inspected') from None
        result = subprocess.run([ctl, '-D', str(data), '-l', str(DEPLOY/'logs/postgres.log'),
                                 '-w', '-t', '45', 'start'], env=env, cwd=ROOT,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError('Project database failed to start; see its postgres.log')
    with psycopg.connect(env['DATABASE_URL'].replace('postgresql+psycopg://','postgresql://'), connect_timeout=3) as conn:
        conn.execute('SELECT 1')


def start():
    if alive(record()):
        status()
        return
    env = environment()
    env.update({'PBPK_RECOMMENDATION_ONLINE_ENABLED':'false', 'CUDA_VISIBLE_DEVICES':'',
                'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2'})
    ensure_database(env)
    key = RUN / 'access.key'
    if not key.exists():
        with key.open('x') as f:
            f.write(secrets.token_urlsafe(24)+'\n')
    key.chmod(0o600)
    # Streamlit remains persistently disabled.
    (DEPLOY/'ui.disabled').touch()
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for port in range(18620,18720):
            try:
                listener.bind(('127.0.0.1',port))
                break
            except OSError:
                continue
        else:
            raise RuntimeError('No free loopback port in workspace range')
        listener.listen(128)
        with (RUN/'web.log').open('ab') as log:
            child = subprocess.Popen([str(PY),'-m','uvicorn','pbpk_kb.web.app:app',
                '--fd',str(listener.fileno()),'--no-access-log','--log-level','warning'],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(listener.fileno(),))
        ticks = (Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
        PID.write_text(json.dumps({'pid':child.pid,'start_ticks':ticks,'port':port,
                                  'url':f'http://127.0.0.1:{port}'},indent=2))
        import httpx
        for _ in range(90):
            if child.poll() is not None:
                raise RuntimeError('Workspace exited; inspect workspace/web.log')
            try:
                r = httpx.get(f'http://127.0.0.1:{port}/health',timeout=1,trust_env=False)
                if r.status_code==200:
                    status()
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.5)
        child.terminate()
        child.wait(timeout=15)
        raise RuntimeError('Workspace startup timed out')


def stop():
    row=record()
    if not alive(row):
        print('Workspace is stopped')
        return
    fd=os.pidfd_open(row['pid'])
    try:
        if not alive(row):
            raise RuntimeError('Project process identity changed')
        signal.pidfd_send_signal(fd,signal.SIGTERM)
        if not select.select([fd],[],[],25)[0]:
            signal.pidfd_send_signal(fd,signal.SIGKILL)
            select.select([fd],[],[],5)
    finally:
        os.close(fd)
    print('Workspace stopped; database and experiments retained')


def status():
    row=record()
    print(json.dumps({'running':alive(row),'url':row.get('url'), 'pid':row.get('pid'),
        'authentication':'personal access code', 'access_code_file':str(RUN/'access.key'),
        'log':str(RUN/'web.log')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    os.umask(0o077)
    RUN.mkdir(parents=True,exist_ok=True)
    RUN.chmod(0o700)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','stop','restart','status'])
    args=parser.parse_args()
    with (RUN/'manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.action in ('stop','restart'):stop()
        if args.action in ('start','restart'):start()
        if args.action=='status':status()
