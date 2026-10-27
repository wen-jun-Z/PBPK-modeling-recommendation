"""Detached, project-only supervisor. Control uses a private project Unix socket.

PID files are informational: stop never sends a signal to a PID read from disk.
Only child processes spawned by this supervisor can be stopped or restarted.
"""
from pathlib import Path
import json,os,signal,socket,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from service_runner import ROOT,DEPLOY,environment
SOCK=DEPLOY/'control.sock';PY=ROOT/'runtime/app-env/bin/python'
CHILDREN={};LOGS={};RESTARTS={};STOP=False

def spawn(key):
    if key=='postgres':
        from pgserver._commands import POSTGRES_BIN_PATH
        command=[str(POSTGRES_BIN_PATH/'postgres'),'-D',str(ROOT/'runtime/postgres-data')]
    else:command=[str(PY),str(ROOT/'scripts/deployment/service_runner.py'),key]
    LOGS[key]=open(DEPLOY/'logs'/f'{key}.log','ab',buffering=0)
    child=subprocess.Popen(command,cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,
        stdout=LOGS[key],stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    CHILDREN[key]=child;(DEPLOY/'pids'/f'{key}.pid').write_text(str(child.pid)+'\n')

def stop_children():
    deadline=time.monotonic()+90
    for key,child in CHILDREN.items():
        if key!='postgres' and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    for key in reversed(list(CHILDREN)):
        child=CHILDREN[key]
        if child.poll() is None:
            child.send_signal(signal.SIGINT if key=='postgres' else signal.SIGTERM)
            try:child.wait(timeout=90 if key=='postgres' else max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                # Database receives a second graceful interrupt, never SIGKILL.
                if key=='postgres':child.send_signal(signal.SIGINT);child.wait(timeout=90)
                else:child.kill();child.wait(timeout=10)
        (DEPLOY/'pids'/f'{key}.pid').unlink(missing_ok=True)
        LOGS[key].close()

def status():
    return {'supervisor_pid':os.getpid(),'parent_pid':os.getppid(),'session_id':os.getsid(0),'ports':json.loads((DEPLOY/'ports.json').read_text()),
        'services':{k:{'pid':p.pid,'running':p.poll() is None,'restart_count':RESTARTS.get(k,0)} for k,p in CHILDREN.items()}}

def terminate(*args):
    global STOP
    STOP=True

def main():
    global STOP
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    os.umask(0o077)
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(SOCK));server.listen(4);server.settimeout(1)
        (DEPLOY/'pids/supervisor.pid').write_text(str(os.getpid())+'\n')
        try:
            spawn('postgres')
            import psycopg
            for _ in range(120):
                if CHILDREN['postgres'].poll() is not None:raise RuntimeError('Project database exited; inspect project postgres.log')
                try:
                    with psycopg.connect(environment()['DATABASE_URL'].replace('postgresql+psycopg://','postgresql://'),connect_timeout=1):pass
                    break
                except psycopg.OperationalError:time.sleep(.5)
            else:raise RuntimeError('Database startup timed out')
            for key in ['embedding','api','ui']:spawn(key)
            while not STOP:
                try:
                    conn,_=server.accept()
                    with conn:
                        conn.settimeout(3);action=conn.recv(1024).decode().strip()
                        if action=='stop':STOP=True
                        conn.sendall(json.dumps(status()).encode())
                except socket.timeout:pass
                if STOP:break
                for key,child in list(CHILDREN.items()):
                    if child.poll() is not None:
                        RESTARTS[key]=RESTARTS.get(key,0)+1
                        if RESTARTS[key]>3:raise RuntimeError(f'{key} repeatedly exited; inspect project logs')
                        LOGS[key].close();time.sleep(2);spawn(key)
                (DEPLOY/'status.json').write_text(json.dumps(status(),indent=2))
        finally:
            stop_children();SOCK.unlink(missing_ok=True)
            (DEPLOY/'pids/supervisor.pid').unlink(missing_ok=True)
            (DEPLOY/'status.json').write_text(json.dumps({'stopped':True}))
if __name__=='__main__':main()
