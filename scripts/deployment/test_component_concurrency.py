"""Three real simultaneous requests: duplicate sharing and distinct GPU waiting."""
from pathlib import Path
import sys,os,json,time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
sys.path.insert(0,str(Path(__file__).resolve().parent))
from service_runner import environment,DEPLOY
os.environ.clear();os.environ.update(environment())
import httpx
base=os.environ['API_BASE_URL'];tag=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
def payload(label):
    return {'compound':'midazolam','objective':'base_pk','route':'oral','constraints':f'Use source-backed components and report evidence gaps. Concurrency verification {tag} {label}.'}
requests=[payload('A'),payload('A'),payload('B')];barrier=Barrier(3)
def send(value):
    barrier.wait(timeout=10);start=time.monotonic()
    r=httpx.post(base+'/recommendation',json=value,timeout=900,trust_env=False)
    return {'http_status':r.status_code,'seconds':round(time.monotonic()-start,2),'response':r.json()}
samples=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    futures=[pool.submit(send,v) for v in requests]
    with httpx.Client(timeout=15,trust_env=False) as client:
        while not all(f.done() for f in futures):
            r=client.get(base+'/recommendation/runtime-status');r.raise_for_status();samples.append(r.json())
            time.sleep(.25)
    results=[f.result() for f in futures]
report={'requests':requests,'results':results,'peak_running':max(x['running'] for x in samples),
    'peak_queued':max(x['queued'] for x in samples),'peak_shared_waiters':max(x['shared_waiters'] for x in samples)}
(DEPLOY/'component_concurrency_test.json').write_text(json.dumps(report,indent=2))
assert all(r['http_status']==200 and r['response']['status']=='recommendation_validated' for r in results),[(r['http_status'],r['response']) for r in results]
ids=[r['response']['recommendation_result_id'] for r in results]
assert ids[0]==ids[1] and ids[0]!=ids[2]
assert report['peak_running']==1 and report['peak_queued']>=1 and report['peak_shared_waiters']>=1
print(json.dumps({'passed':True,'http_statuses':[r['http_status'] for r in results],
    'seconds':[r['seconds'] for r in results],'duplicate_result_id_shared':True,
    **{k:v for k,v in report.items() if k.startswith('peak_')}},indent=2))
