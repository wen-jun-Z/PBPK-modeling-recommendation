"""Real PBPK API + Streamlit tests. No mocked backend or fabricated records."""
from pathlib import Path
import argparse,json,os,sys,time,traceback,zipfile,io
sys.path.insert(0,str(Path(__file__).resolve().parent))
from service_runner import environment,ROOT,DEPLOY
os.environ.clear();os.environ.update(environment())
import httpx
P=argparse.ArgumentParser();P.add_argument('--mode',choices=['api','pages','all'],default='all');args=P.parse_args()
client=httpx.Client(base_url=os.environ['API_BASE_URL'],timeout=120)
results=[]
def call(method,path,**kw):
    r=client.request(method,path,**kw);r.raise_for_status();return r.json()
def check(name,fn):
    t=time.time()
    try:details=fn();r={'test':name,'status':'passed','seconds':round(time.time()-t,2),'details':details}
    except Exception as e:
        # Own database password only; never emit configuration/environment values.
        msg=str(e).replace((DEPLOY/'postgres-password').read_text().strip(),'<redacted>')
        r={'test':name,'status':'failed','seconds':round(time.time()-t,2),'error':msg[:1800]}
    results.append(r);print(json.dumps(r,ensure_ascii=False,default=str),flush=True)
    (DEPLOY/f'smoke_{args.mode}.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str))
def api_e2e():
    compounds=call('GET','/compounds/search',params={'q':'midazolam'});assert compounds
    compound=compounds[0]
    evidence=call('GET','/evidence/search',params={'compound':'midazolam','q':'clearance','limit':10});assert evidence
    e=evidence[0];detail=call('GET',f'/evidence/{e["evidence_type"]}/{e["evidence_id"]}')
    assert detail['summary'] and e['source_location']
    semantic=call('GET','/search/semantic',params={'q':'midazolam clearance','limit':5});assert semantic
    validated=call('GET','/recommendation/validated',params={'compound':'midazolam','limit':100});assert validated
    req=next((r['request_json'] for r in validated if r['objective']=='base_pk'),validated[0]['request_json'])
    components=call('POST','/recommendation',json=req)
    assert components['status']=='recommendation_validated' and components['citations']
    ranked=call('POST','/recommendation/public-model-ranking',json={'compound':'Rosuvastatin','objective':'public_model_reuse','species':'human','route':'oral','dose':10,'dose_unit':'mg','single_or_multiple':'single'})
    assert ranked.get('candidates') or ranked.get('same_compound_candidates')
    qa=call('POST','/qa/ask',json={'question':'What is known about midazolam clearance?','compound':'midazolam','evidence_limit':5,'chunk_limit':3,'model_limit':3})
    assert qa['citations'] and qa['key_evidence'] and qa['answer']
    return {'compound':compound['preferred_name'],'evidence_hits':len(evidence),'provenance_verified':True,'semantic_hits':len(semantic),'component_status':components['status'],'component_citations':len(components['citations']),'model_status':ranked['status'],'model_candidates':len(ranked.get('candidates',[])),'qa_citations':len(qa['citations'])}
def exports():
    outputs=[]
    for route,body in [('evidence',{'compound':'midazolam','limit':20}),('compound-profile',{'compound':'midazolam'}),('model-registry',{}),('stage1-summary',{})]:
        result=call('POST','/exports/'+route,json={**body,'requested_by':'pbpk_deployment_smoke'})
        path=Path(result['output_path']);assert path.resolve().is_relative_to(ROOT) and path.is_file() and path.stat().st_size>30
        outputs.append({'kind':route,'bytes':path.stat().st_size})
    return outputs
def blueprint():
    registered=call('GET','/blueprints')
    if not registered:
        return {'registered_blueprints':0,'optional_check':'not_applicable'}
    blueprint_id=registered[0]['blueprint_id']
    d=call('GET','/blueprints/'+blueprint_id)
    counts={}
    for sub in ['versions','parameters','components','artifacts','simulation-runs']:
        rows=call('GET',f'/blueprints/{blueprint_id}/{sub}');assert rows;counts[sub]=len(rows)
    r=client.get(f'/blueprints/{blueprint_id}/package');r.raise_for_status();assert zipfile.is_zipfile(io.BytesIO(r.content))
    validation=call('POST',f'/blueprints/{blueprint_id}/validate')
    compilation=call('POST',f'/blueprints/{blueprint_id}/compile')
    simulation=call('POST',f'/blueprints/{blueprint_id}/run')
    assert validation['status']=='deterministic_validation_completed'
    assert compilation['job']['status']=='succeeded' and simulation['runs']
    return {'scientific_status':d.get('scientific_status'),'counts':counts,'download_bytes':len(r.content),'validation':validation['status'],'compilation':compilation['status'],'simulation':simulation['status']}
if args.mode in ['api','all']:
    check('knowledge_to_components_and_model_e2e',api_e2e)
    check('all_four_exports',exports)
    check('blueprint_details_and_download',blueprint)
if args.mode in ['pages','all']:
    from streamlit.testing.v1 import AppTest
    def ok(app):
        assert not app.exception,[e.message for e in app.exception]
        assert not app.error,[e.value for e in app.error]
        return {'tables':len(app.dataframe),'metrics':len(app.metric),'titles':[x.value for x in app.title]}
    pages=[ROOT/'src/pbpk_kb/ui/streamlit_app.py',*sorted((ROOT/'src/pbpk_kb/ui/pages').glob('*.py'))]
    for page in pages:
        check('page:'+page.name,lambda page=page:ok(AppTest.from_file(str(page)).run(timeout=120)))
    def qa_page():
        a=AppTest.from_file(str(ROOT/'src/pbpk_kb/ui/pages/09_qa_assistant.py')).run(timeout=120)
        next(b for b in a.button if b.label=='Ask').click().run(timeout=120)
        d=ok(a);assert len(a.dataframe)>=3;return d
    check('interaction:qa_submit',qa_page)
    def rank_page():
        a=AppTest.from_file(str(ROOT/'src/pbpk_kb/ui/pages/14_recommendation_agent.py')).run(timeout=120)
        next(b for b in a.button if b.label=='Rank downloaded public models').click().run(timeout=120)
        d=ok(a);assert a.session_state['public_model_ranking_result']['candidates'];return d
    check('interaction:public_model_ranking',rank_page)
    def export_page():
        a=AppTest.from_file(str(ROOT/'src/pbpk_kb/ui/pages/10_exports.py')).run(timeout=120)
        next(b for b in a.button if b.label=='Export evidence CSV').click().run(timeout=120)
        d=ok(a);assert len(a.get('download_button'))>=1;return d
    check('interaction:evidence_csv_download',export_page)
    def search_modes():
        a=AppTest.from_file(str(ROOT/'src/pbpk_kb/ui/pages/05_evidence_search.py')).run(timeout=120)
        next(x for x in a.text_input if x.label=='Parameter or text').set_value('clearance').run(timeout=120)
        counts={}
        for mode in ['Structured','Semantic','Hybrid']:
            next(x for x in a.radio if x.label=='Search mode').set_value(mode).run(timeout=120)
            ok(a);counts[mode]=len(a.dataframe)
            assert counts[mode]==(2 if mode=='Hybrid' else 1)
            assert all(len(d.value)>0 for d in a.dataframe)
        return counts
    check('interaction:all_search_modes_and_filter',search_modes)
    def graph_modes():
        a=AppTest.from_file(str(ROOT/'src/pbpk_kb/ui/pages/12_knowledge_graph.py')).run(timeout=120)
        counts={}
        for mode in ['Compound','Enzyme / transporter','Source','Public model']:
            next(x for x in a.radio if x.label=='Graph center').set_value(mode).run(timeout=120)
            ok(a);counts[mode]=next(x.value for x in a.metric if x.label=='Nodes')
            assert int(counts[mode])>0 and len(a.get('download_button'))>=1
        return counts
    check('interaction:all_graph_centers',graph_modes)
    def blueprint_page():
        if not call('GET','/blueprints'):
            return {'registered_blueprints':0,'optional_check':'not_applicable'}
        a=AppTest.from_file(str(ROOT/'src/pbpk_kb/ui/pages/14_recommendation_agent.py')).run(timeout=120)
        target=next(x for x in a.selectbox if x.label=='Target compound')
        target.set_value(next(x for x in target.options if x.lower()=='caffeine')).run(timeout=120)
        next(b for b in a.button if b.label=='Rank downloaded public models').click().run(timeout=120)
        next(b for b in a.button if b.label=='Check registered executable blueprint').click().run(timeout=120)
        assert not a.exception
        assert len(a.error)==1 and a.error[0].value.startswith('Engineering execution succeeded, but scientific registration is blocked')
        d={'scientific_gate_message_verified':True,'tables':len(a.dataframe)}
        assert len(a.get('download_button'))>=1
        assert any(m.value=='absorption_structure_failed' for m in a.metric)
        return d
    check('interaction:blueprint_workspace_and_package',blueprint_page)
failures=[x for x in results if x['status']=='failed']
print(json.dumps({'passed':len(results)-len(failures),'failed':len(failures)},ensure_ascii=False),flush=True)
sys.exit(bool(failures))
