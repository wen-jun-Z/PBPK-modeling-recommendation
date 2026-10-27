"""Enable/disable only this project's Streamlit, without restarting its backend."""
from pathlib import Path
import argparse,json,os,select,signal,socket,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from services import ROOT,DEPLOY,request

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','stop'])
    action=parser.parse_args().action
    marker=DEPLOY/'ui.disabled'
    if action=='start':
        marker.unlink(missing_ok=True)
        print('Streamlit enabled; the project runner will start it.')
        return
    marker.touch(mode=0o600,exist_ok=True)
    state=request('status');pid=state['services']['ui']['pid']
    before={k:v['pid'] for k,v in state['services'].items() if k!='ui'}
    # A pidfd prevents accidentally signaling a subsequently reused PID.
    try:fd=os.pidfd_open(pid)
    except ProcessLookupError:fd=None
    if fd is not None:
        try:
            proc=Path('/proc')/str(pid)
            if proc.stat().st_uid!=os.getuid():raise RuntimeError('UI process owner mismatch')
            status=dict(line.split(':',1) for line in (proc/'status').read_text().splitlines() if ':' in line)
            if int(status['PPid'])!=state['supervisor_pid']:raise RuntimeError('UI parent mismatch')
            args=(proc/'cmdline').read_bytes().decode().split('\0')
            if str(ROOT/'src/pbpk_kb/ui/streamlit_app.py') in args and 'streamlit' in args:
                signal.pidfd_send_signal(fd,signal.SIGTERM)
                if not select.select([fd],[],[],10)[0]:
                    signal.pidfd_send_signal(fd,signal.SIGKILL)
                    if not select.select([fd],[],[],10)[0]:raise RuntimeError('UI process has not exited')
            elif str(ROOT/'scripts/deployment/service_runner.py') not in args:
                raise RuntimeError('PID is not a verified project UI process')
        finally:os.close(fd)
    # Only bind the project's intended port; never identify another listener.
    port=state['ports']['ui']
    for _ in range(30):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:probe.bind(('127.0.0.1',port));break
            except OSError:time.sleep(.5)
    else:raise RuntimeError('UI port could not be confirmed closed')
    after=request('status')
    assert all(after['services'][k]['pid']==v and after['services'][k]['running'] for k,v in before.items())
    result={'streamlit_stopped':True,'disabled_persistently':True,'port_released':port,'backend_pids_unchanged':before}
    (DEPLOY/'ui_stopped.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
