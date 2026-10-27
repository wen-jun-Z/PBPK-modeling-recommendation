"""Run one PBPK service with a clean, project-owned environment."""
from pathlib import Path
import json,os,sys,time
from dotenv import dotenv_values
ROOT=Path(__file__).resolve().parents[2]
DEPLOY=ROOT/'runtime/deployment'

def environment():
    values={k:v for k,v in dotenv_values(DEPLOY/'app.env').items() if v is not None}
    values.update({'HOME':str(DEPLOY/'home'),'PATH':str(ROOT/'runtime/app-env/bin')+':/usr/bin:/bin',
        'LANG':'C.UTF-8','PYTHONPATH':str(ROOT/'src'),'PBPK_ENV_FILE':str(DEPLOY/'app.env'),
        'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','XDG_CACHE_HOME':str(DEPLOY/'cache'),
        'TMPDIR':str(DEPLOY/'cache'),'NO_PROXY':'127.0.0.1,localhost'})
    return values

def main():
    key=sys.argv[1]
    ports=json.loads((DEPLOY/'ports.json').read_text())
    py=str(ROOT/'runtime/app-env/bin/python')
    commands={
        'api':[py,'-m','uvicorn','pbpk_kb.api.main:app','--host','127.0.0.1','--port',str(ports['api']),'--log-level','warning'],
        'embedding':[py,'-m','uvicorn','pbpk_kb.embeddings.service:app','--host','127.0.0.1','--port',str(ports['embedding']),'--log-level','warning'],
        'ui':[py,'-m','streamlit','run',str(ROOT/'src/pbpk_kb/ui/streamlit_app.py'),'--server.address=127.0.0.1',f'--server.port={ports["ui"]}','--server.headless=true','--server.fileWatcherType=none','--browser.gatherUsageStats=false'],
    }
    (DEPLOY/'pids'/f'{key}.pid').write_text(str(os.getpid())+'\n')
    os.chdir(ROOT)
    # Keep the supervisor's slot alive without launching Streamlit while disabled.
    while key=='ui' and (DEPLOY/'ui.disabled').exists():
        time.sleep(1)
    os.execve(py,commands[key],environment())
if __name__=='__main__':main()
