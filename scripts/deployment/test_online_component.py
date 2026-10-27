from pathlib import Path
from datetime import datetime, timezone
import os,sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from service_runner import environment,DEPLOY
os.environ.clear();os.environ.update(environment())
import httpx
payload={'compound':'midazolam','objective':'base_pk','route':'oral','constraints':'Report source-backed components and evidence gaps for deployment verification.'}
start=time.time()
try:
 r=httpx.post(os.environ['API_BASE_URL']+'/recommendation',json=payload,timeout=1800)
 result={'http_status':r.status_code,'elapsed_seconds':round(time.time()-start,2),'request':payload,'response':r.json()}
except Exception as e:result={'exception_type':type(e).__name__,'elapsed_seconds':round(time.time()-start,2)}
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
serialized=json.dumps(result,indent=2)
(DEPLOY/f'online_component_test_{stamp}.json').write_text(serialized)
canonical=DEPLOY/'online_component_test.json'
if not canonical.exists() or json.loads(canonical.read_text()).get('http_status')!=200:
 canonical.write_text(serialized)
print(json.dumps({k:v for k,v in result.items() if k!='response'}),flush=True)
if 'response' in result:print('Response status:',result['response'].get('status'),'fields:',list(result['response']))
response=result.get('response',{})
sys.exit(0 if result.get('http_status')==200 and response.get('status')=='recommendation_validated' and response.get('citations') and response.get('component_recommendations') else 1)
