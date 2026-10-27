"""Verify CUDA embedding and an uncached, end-to-end component request."""
from pathlib import Path
import json,os,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from service_runner import DEPLOY,environment
os.environ.clear();os.environ.update(environment())
import httpx
from sqlalchemy import create_engine,text
from pbpk_kb.api.recommendation_routes import _request_sha256
from pbpk_kb.recommendation.online_agent import _runtime_id
from pbpk_kb.recommendation.schemas import RecommendationRequest

payload={'compound':'midazolam','objective':'base_pk','route':'oral',
    'constraints':'Use source-backed evidence and explicitly report missing components; GPU deployment verification.'}
request=RecommendationRequest.model_validate(payload)
with create_engine(os.environ['DATABASE_URL']).connect() as c:
    cached=c.execute(text('SELECT count(*) FROM recommendation_result WHERE request_sha256=:h AND status=:s'),
        {'h':_request_sha256(request),'s':'recommendation_validated'}).scalar_one()
with httpx.Client(timeout=900,trust_env=False) as client:
    embedding=client.post(os.environ['PBPK_EMBEDDING_API_URL']+'/embed',json={'texts':['midazolam clearance']})
    embedding.raise_for_status();ev=embedding.json()
    assert ev['device'].startswith('cuda') and len(ev['embeddings'][0])==768
    start=time.monotonic();response=client.post(os.environ['API_BASE_URL']+'/recommendation',json=payload)
    result={'http_status':response.status_code,'elapsed_seconds':round(time.monotonic()-start,2),
        'cached_before_request':bool(cached),'request':payload,'response':response.json(),
        'embedding':{k:ev[k] for k in ['device','model','dimension']}}
    run=Path(os.environ['PBPK_RECOMMENDATION_ONLINE_OUTPUT_DIR'])/_runtime_id(request)
    manifest=run/'recommendation_run_manifest.json'
    if manifest.is_file():result['run_manifest']=json.loads(manifest.read_text())
    output=DEPLOY/('gpu_serving_cached_check.json' if cached else 'gpu_serving_test.json')
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['response','run_manifest']},indent=2),flush=True)
    response.raise_for_status()
    assert result['response']['status']=='recommendation_validated' and result['response']['citations']
    runtime=result['run_manifest']['runtime'];assert runtime['model_device'].startswith('cuda')
    print(json.dumps({'runtime':runtime,'components':len(result['response']['component_recommendations']),
        'citations':len(result['response']['citations']),'status':result['response']['status']},indent=2),flush=True)
