"""Start/stop/status for the detached PBPK project services; no global changes."""
from pathlib import Path
import argparse,fcntl,json,os,socket,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2];DEPLOY=ROOT/'runtime/deployment'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from service_runner import environment
SOCK=DEPLOY/'control.sock';PY=ROOT/'runtime/app-env/bin/python';KEYS=['postgres','embedding','api','ui']
def request(action):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(5);s.connect(str(SOCK));s.sendall(action.encode());return json.loads(s.recv(65536))
def health(result):
    import httpx,psycopg
    checks={};ports=result['ports']
    ui_disabled=(DEPLOY/'ui.disabled').exists()
    if ui_disabled and 'ui' in result['services']:
        result['services']['ui'].update({'enabled':False,'runner_alive':result['services']['ui']['running'],'running':False})
    for key,path in [('api','/health'),('embedding','/health'),('ui','/_stcore/health')]:
        if key=='ui' and ui_disabled:
            checks[key]=None
            continue
        try:
            r=httpx.get(f'http://127.0.0.1:{ports[key]}{path}',timeout=10,trust_env=False)
            checks[key]=r.status_code==200
        except httpx.HTTPError:checks[key]=False
    try:
        with psycopg.connect(environment()['DATABASE_URL'].replace('postgresql+psycopg://','postgresql://'),connect_timeout=3) as c:
            checks['postgres']=c.execute('SELECT 1').fetchone()[0]==1
    except psycopg.Error:checks['postgres']=False
    result['health']=checks
    result['healthy']=all(v for v in checks.values() if v is not None) and all(v['running'] for v in result['services'].values() if v.get('enabled',True))
    return result
def owned_systemd_transition():
    names=[]
    for key in KEYS:
        name=f'pbpk-recommendation-{key}.service'
        r=subprocess.run(['systemctl','--user','show',name,'--property=ExecStart','--value'],capture_output=True,text=True)
        if not r.stdout.strip():continue
        if str(ROOT) not in r.stdout:raise RuntimeError(f'{name} is not verified as a project unit')
        names.append(name)
    if names:
        subprocess.run(['systemctl','--user','stop',*reversed(names)],check=True)
        subprocess.run(['systemctl','--user','disable',*names],check=True,capture_output=True)
def choose_ports():
    ports=json.loads((DEPLOY/'ports.json').read_text());sockets=[];used=set()
    try:
        for key in KEYS:
            start=ports[key]
            for port in range(start,start+150):
                if port in used:continue
                s=socket.socket()
                s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                try:s.bind(('127.0.0.1',port))
                except OSError:s.close();continue
                sockets.append(s);ports[key]=port;used.add(port);break
            else:raise RuntimeError('No available project port')
        (DEPLOY/'ports.json').write_text(json.dumps(ports,indent=2))
        from dotenv import dotenv_values
        values={k:v for k,v in dotenv_values(DEPLOY/'app.env').items() if v is not None}
        values['API_BASE_URL']=f'http://127.0.0.1:{ports["api"]}'
        values['PBPK_EMBEDDING_API_URL']=f'http://127.0.0.1:{ports["embedding"]}'
        values['DATABASE_URL']=values['DATABASE_URL'].rsplit('@',1)[0]+f'@127.0.0.1:{ports["postgres"]}/pbpk_kb'
        (DEPLOY/'app.env').write_text('\n'.join(k+'='+v for k,v in values.items())+'\n')
        config=ROOT/'runtime/postgres-data/postgresql.conf'
        with config.open('a') as f:
            f.write(f"\n# Project deployment: loopback TCP, independent of login runtime directories\nport = {ports['postgres']}\nunix_socket_directories = ''\nlisten_addresses = '127.0.0.1'\n")
    finally:
        for s in sockets:s.close()
def stop():
    try:request('stop')
    except (FileNotFoundError,ConnectionRefusedError):return
    except (TimeoutError,ConnectionResetError):pass  # Supervisor may already be finishing its children.
    for _ in range(660):
        if not SOCK.exists():return
        time.sleep(.5)
    raise RuntimeError('Project shutdown still in progress; inspect project log')
def start():
    try:
        result=request('status')
        print(json.dumps(result,indent=2));return
    except (FileNotFoundError,ConnectionRefusedError):pass
    owned_systemd_transition()
    if SOCK.exists():SOCK.unlink()
    choose_ports()
    with (DEPLOY/'logs/supervisor.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([str(PY),str(ROOT/'scripts/deployment/supervisor.py')],cwd=ROOT,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if p.poll() is not None:raise RuntimeError('Project supervisor failed; inspect supervisor.log')
        try:
            result=request('status')
            if len(result['services'])==4 and all(v['running'] for v in result['services'].values()):
                result=health(result)
                if result['healthy']:print(json.dumps(result,indent=2));return
        except (FileNotFoundError,ConnectionRefusedError,TimeoutError):pass
        time.sleep(.5)
    raise RuntimeError('Project startup readiness exceeded 300 seconds')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['start','stop','restart','status']);a=p.parse_args()
    with (DEPLOY/'manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if a.action in ['stop','restart']:stop()
        if a.action in ['start','restart']:start()
        if a.action=='status':
            try:
                result=health(request('status'));print(json.dumps(result,indent=2));sys.exit(0 if result['healthy'] else 1)
            except (FileNotFoundError,ConnectionRefusedError):print(json.dumps({'running':False}));sys.exit(1)
if __name__=='__main__':main()
