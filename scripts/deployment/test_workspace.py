"""Real API and Chromium acceptance for this project's private web workspace.

--results-only checks preserved research APIs without launching Chromium.
"""
import json
import re
import os
import sys
from pathlib import Path
from urllib.parse import urljoin

from service_runner import ROOT, DEPLOY, environment

env=environment()
os.environ.clear()
os.environ.update(env)
os.environ['PLAYWRIGHT_BROWSERS_PATH']=str(DEPLOY/'cache/playwright')
os.environ['LD_LIBRARY_PATH']=str(DEPLOY/'cache/browser-libs/extracted/usr/lib/x86_64-linux-gnu')

import httpx
from playwright.sync_api import sync_playwright, expect

RUN=DEPLOY/'workspace'
BASE=json.loads((RUN/'pid.json').read_text())['url']
KEY=(RUN/'access.key').read_text().strip()
results=[]
COMPONENTS_ONLY='--components-only' in sys.argv[1:]
GRAPH_ONLY='--graph-only' in sys.argv[1:]
BLUEPRINTS_ONLY='--blueprints-only' in sys.argv[1:]
RESEARCH_API_ONLY='--results-only' in sys.argv[1:]
FOCUSED=COMPONENTS_ONLY or GRAPH_ONLY or BLUEPRINTS_ONLY or RESEARCH_API_ONLY

def record(name, **details):
    results.append({'check':name,'status':'passed',**details})
    print(name, json.dumps(details,ensure_ascii=False),flush=True)
    report='graph_acceptance.json' if GRAPH_ONLY else 'component_acceptance.json' if COMPONENTS_ONLY else 'blueprint_acceptance.json' if BLUEPRINTS_ONLY else 'research_acceptance.json' if RESEARCH_API_ONLY else 'acceptance.json'
    (RUN/report).write_text(json.dumps(results,ensure_ascii=False,indent=2))

def check_saved_components(page):
    response=page.request.get(BASE+'/api/recommendation/validated?limit=500')
    assert response.status==200
    saved=response.json()
    assert saved, 'No saved component recommendations returned'
    page.locator('nav a[href="#recommend"]').click()
    page.get_by_role('button',name='组件推荐',exact=True).click()
    form=page.locator('#component-form')
    form.locator('[name="compound"]').select_option('midazolam')
    form.locator('[name="objective"]').select_option('base_pk')
    form.locator('[name="route"]').select_option('oral')
    form.get_by_role('button',name='查找组件推荐 →').click()
    expected=[r for r in saved if r['compound_name'].lower()=='midazolam'
              and r['objective']=='base_pk' and r['request_json'].get('route')=='oral']
    buttons=page.locator('#main button[data-action="component-detail"]')
    expect(buttons).to_have_count(len(expected))
    assert set(buttons.evaluate_all('(nodes)=>nodes.map(n=>n.dataset.id)'))=={
        r['recommendation_result_id'] for r in expected}
    expect(page.locator('#main thead')).to_contain_text('适用场景')
    expect(page.locator('#main thead')).to_contain_text('保存时间')
    for r in expected:
        row=page.locator(f'#main tr:has(button[data-id="{r["recommendation_result_id"]}"])')
        expect(row).to_contain_text('口服')
        expect(row).to_contain_text('人类')
        expect(row).to_contain_text(r['created_at'][:19].replace('T',' ')+' UTC')
    record('browser_component_request_filters_and_saved_scenarios',matched=len(expected))

    # Inspect all preserved records, including the two similarly named oral requests.
    for field in ('compound','objective','route'):
        form.locator(f'[name="{field}"]').select_option('')
    form.get_by_role('button',name='查找组件推荐 →').click()
    expect(buttons).to_have_count(len(saved))
    names={'absorption':'吸收','distribution':'分布','excretion':'排泄与清除',
           'metabolism':'代谢','physicochemical':'理化性质','ddi':'药物相互作用',
           'special_population':'特殊人群','public_model_template':'公共模型模板','transport':'转运'}
    cards_checked=parameters_checked=0
    for r in saved:
        rid=r['recommendation_result_id']
        response=page.request.get(BASE+'/api/recommendation/validated/'+rid)
        assert response.status==200
        d=response.json()['recommendation_json']
        page.locator(f'#main button[data-action="component-detail"][data-id="{rid}"]').click()
        expect(page.locator('#detail')).to_be_visible()
        body=page.locator('#detail-body')
        expect(body.locator('.component-card')).to_have_count(len(d['component_recommendations']))
        expect(body).to_contain_text('尚未完成校准或仿真验证')
        expect(body.locator('pre')).to_have_count(0)
        assert '[object Object]' not in body.inner_text()
        structure=d.get('recommended_model_structure') or {}
        if not structure.get('required_compartments'):
            assert '涉及房室' not in body.locator('dt').all_text_contents()
        if not d.get('metabolism_pathways'):
            assert '涉及代谢酶' not in body.locator('dt').all_text_contents()
        if not d.get('transport_pathways'):
            assert '涉及转运体' not in body.locator('dt').all_text_contents()
        for component in d['component_recommendations']:
            card=body.locator(f'.component-card[data-component="{component["component"]}"]')
            expect(card.locator('header h3')).to_have_text(names[component['component']])
            refs=[x for x in d.get('citations',[]) if x['evidence_id'] in component['evidence_ids']]
            card.locator('summary').click()
            expected_links=[f'/api/workspace/evidence/{x["evidence_type"]}/{x["evidence_id"]}/source'
                            for x in refs if x.get('evidence_type')]
            links=card.get_by_role('link',name='查看来源 ↗')
            expect(links).to_have_count(len(expected_links))
            assert links.evaluate_all('(nodes)=>nodes.map(n=>n.getAttribute("href"))')==expected_links
            for ref in refs:
                if ref.get('supporting_text'):
                    expect(card).to_contain_text(ref['supporting_text'])
            for link in links.all():
                expect(link).to_have_attribute('target','_blank')
            cards_checked+=1
        parameters=d.get('parameter_candidates',[])
        section=body.locator('section.component-section').filter(
            has=page.get_by_role('heading',name='参数候选',exact=True))
        expect(section.locator('tbody tr')).to_have_count(len(parameters))
        for i,p in enumerate(parameters):
            row=section.locator('tbody tr').nth(i)
            expect(row.locator('td').nth(0)).to_contain_text(re.compile(r'\S'))
            for key in ('value','value_low','value_high'):
                if p.get(key) is not None:
                    value=format(p[key],'.15g') if isinstance(p[key],(int,float)) else str(p[key])
                    expect(row.locator('td').nth(1)).to_contain_text(value)
            if p.get('canonical_unit'):
                expect(row.locator('td').nth(1)).to_contain_text(p['canonical_unit'])
            context=p.get('study_context') or {}
            if context.get('condition'):
                expect(row.locator('td').nth(2)).to_contain_text(context['condition'])
            if p.get('population') or context.get('population'):
                expect(row.locator('td').nth(2)).to_contain_text(p.get('population') or context['population'])
            parameters_checked+=1
        # Rejected numeric bindings must not silently reappear as model parameters.
        assert len(parameters)==section.locator('tbody tr').count()
        page.get_by_role('button',name='关闭详情').click()
    record('browser_saved_component_details_and_evidence_bindings',records=len(saved),
           component_cards=cards_checked,parameters=parameters_checked)

def check_source_link(page, link, expected_location=None):
    expect(link).to_have_text('查看来源 ↗')
    expect(link).to_have_attribute('target', '_blank')
    expect(link).to_have_attribute(
        'href', re.compile(r'^/api/workspace/evidence/[a-z_]+/[0-9a-f-]{36}/source$')
    )
    href=link.get_attribute('href')
    response=page.request.get(urljoin(BASE, href), max_redirects=0)
    assert response.status==307
    location=response.headers.get('location', '')
    assert location and 'efetch' not in location
    if expected_location is not None:
        assert location==expected_location
    destination=urljoin(BASE, location)
    # Validate navigation itself; third-party page contents are not under our control.
    with page.expect_popup() as opened:
        link.click()
    popup=opened.value
    expect(popup).to_have_url(destination)
    expect(page.locator('#detail')).to_be_hidden()
    popup.close()
    return location

def check_graph(page):
    """Exercise real graph data, readable geometry and navigation without GPU work."""
    compounds=page.request.get(BASE+'/api/compounds/search?limit=250').json()
    samples=[next(c for c in compounds if c['preferred_name']==name)
             for name in ('caffeine','midazolam','metformin')]
    page.locator('nav a[href="#graph"]').click()
    def check_geometry():
        # SVG bounding boxes check the labels themselves, not just node centres.
        geometry=page.locator('.graph-svg').evaluate('''svg=>{
          const rows=[...svg.querySelectorAll('.graph-node')].map(node=>{
            const rect=node.querySelector('rect').getBoundingClientRect();
            const names=[...node.querySelectorAll('text')].map(t=>{
              const r=t.getBoundingClientRect();return {text:t.textContent,
                inside:r.left>=rect.left-.5&&r.right<=rect.right+.5&&
                       r.top>=rect.top-.5&&r.bottom<=rect.bottom+.5};});
            return {id:node.dataset.node,left:rect.left,right:rect.right,
                    top:rect.top,bottom:rect.bottom,names};});
          const overlaps=[];
          rows.forEach((a,i)=>rows.slice(i+1).forEach(b=>{
            if(Math.min(a.right,b.right)>Math.max(a.left,b.left)+.5&&
               Math.min(a.bottom,b.bottom)>Math.max(a.top,b.top)+.5)
              overlaps.push([a.id,b.id]);}));
          return {overlaps,overflow:rows.flatMap(n=>n.names.filter(t=>!t.inside).map(t=>({id:n.id,...t})))};
        }''')
        assert not geometry['overlaps'],geometry
        assert not geometry['overflow'],geometry

    graphs=[]
    for compound in samples:
        cid=compound['compound_id']
        response=page.request.get(BASE+'/api/graph/compound/'+cid)
        assert response.status==200
        graph=response.json();graphs.append(graph)
        expected={n['id']:n for n in graph['nodes']}
        evidence=[n for n in graph['nodes'] if n['type']=='evidence']
        models=[n for n in graph['nodes'] if n['type']=='model']
        targets=[n for n in graph['nodes'] if n['type']=='protein_target']
        sources=[n for n in graph['nodes'] if n['type']=='source']
        assert len(evidence)==compound['evidence_count']
        page.locator('#graph-form select').select_option(cid)
        page.locator('#graph-form button').click()
        expect(page.locator('.graph-inspector h2')).to_have_text(compound['preferred_name'])
        expect(page.locator('.graph-caption')).to_contain_text(f'{len(evidence)} 条证据')
        expect(page.locator('.graph-caption')).to_contain_text(f'{len(models)} 个登记模型')
        expect(page.locator('.graph-caption')).to_contain_text(f'{len(targets)} 个酶/转运体')
        expect(page.locator('.graph-node.source')).to_have_count(len(sources))
        categories={n['properties']['evidence_type'] for n in evidence}
        expect(page.locator('.graph-node.evidence')).to_have_count(len(categories))
        for kind,members in (('model',models),('protein_target',targets)):
            group=page.locator(f'.graph-node.{kind}')
            expect(group).to_have_count(int(bool(members)))
            if members:
                expect(group.locator('.node-meta')).to_contain_text(str(len(members)))
                group.click()
                records=page.locator('.graph-inspector .graph-records button')
                expect(records).to_have_count(len(members))
                assert set(records.evaluate_all('(nodes)=>nodes.map(n=>n.dataset.graphNode)'))==set(n['id'] for n in members)
                records.first.click()
                expect(page.locator('.graph-inspector h2')).to_have_text(members[0]['label'])

        check_geometry()
        edge_titles=page.locator('.graph-edge title').all_text_contents()
        assert sum(int(re.search(r' · (\d+) 条关联记录$', t).group(1)) for t in edge_titles)==len(graph['edges'])

        # Classification and pagination must retain every original evidence ID.
        visited=[]
        for category in sorted(categories):
            page.locator('[data-graph-action="overview"]').click()
            members=[n for n in evidence if n['properties']['evidence_type']==category]
            node=page.locator(f'.graph-node[data-node="group:evidence:{category}"]')
            node.click()
            expect(page.locator('.graph-inspector .graph-description')).to_have_text(f'{len(members)} 条证据')
            expect(page.locator('.graph-records button')).to_have_count(min(6,len(members)))
            page.locator('[data-graph-action="expand"]').click()
            expect(page.locator('.graph-evidence-controls select')).to_have_value(category)
            category_seen=[]
            while True:
                evidence_nodes=page.locator('.graph-node.evidence')
                ids=evidence_nodes.evaluate_all('(nodes)=>nodes.map(n=>n.dataset.node)')
                assert 0<len(ids)<=8
                check_geometry()
                assert set(ids)<=set(n['id'] for n in members)
                assert not set(ids)&set(category_seen)
                category_seen.extend(ids)
                if len(category_seen)<=8:
                    evidence_nodes.first.click()
                    original=expected[ids[0]]['properties']
                    link=page.locator('.graph-inspector .graph-source-link')
                    expect(link).to_have_attribute('href',
                        f'/api/workspace/evidence/{original["evidence_type"]}/{original["evidence_id"]}/source')
                    expect(link).to_have_attribute('target','_blank')
                    expect(page.locator('.graph-excerpt')).to_be_visible()
                next_button=page.locator('[data-graph-action="next"]')
                if next_button.is_disabled():break
                next_button.click()
            assert len(category_seen)==len(members)
            assert set(category_seen)==set(n['id'] for n in members)
            visited.extend(category_seen)
        assert len(visited)==len(evidence)==len(set(visited))
        select=page.locator('.graph-evidence-controls select')
        select.select_option('')
        expect(page.locator('.graph-page-status')).to_contain_text(f'1–8 / {len(evidence)} 条')
        expect(page.locator('[data-graph-action="prev"]')).to_be_disabled()
        page.locator('[data-graph-action="next"]').click()
        expect(page.locator('.graph-page-status')).to_contain_text('9–16')
        page.locator('[data-graph-action="prev"]').click()
        expect(page.locator('.graph-page-status')).to_contain_text('1–8')
        record('graph_full_counts_geometry_and_evidence_navigation',compound=compound['preferred_name'],
               evidence=len(evidence),models=len(models),targets=len(targets),nodes=len(graph['nodes']),edges=len(graph['edges']))

    page.locator('[data-graph-action="overview"]').click()
    svg=page.locator('.graph-svg')
    initial=svg.get_attribute('viewBox')
    page.get_by_role('button',name='放大图谱',exact=True).click()
    expect(page.locator('.graph-zoom')).to_have_text('125%')
    assert svg.get_attribute('viewBox')!=initial
    page.get_by_role('button',name='缩小图谱',exact=True).click()
    expect(page.locator('.graph-zoom')).to_have_text('100%')
    svg.focus();svg.press('+')
    expect(page.locator('.graph-zoom')).to_have_text('125%')
    svg.press('0')
    assert svg.get_attribute('viewBox')==initial
    box=svg.bounding_box()
    page.mouse.move(box['x']+box['width']*.45,box['y']+box['height']*.08)
    page.mouse.wheel(0,-100)
    expect(page.locator('.graph-zoom')).to_have_text('112%')
    page.locator('[data-graph-action="reset"]').click()
    # Drag the gap between columns; dragging a node intentionally selects it.
    start=svg.evaluate('''svg=>{const p=svg.createSVGPoint();p.x=280;p.y=60;
      const q=p.matrixTransform(svg.getScreenCTM());return {x:q.x,y:q.y};}''')
    page.mouse.move(start['x'],start['y'])
    page.mouse.down()
    page.mouse.move(start['x']+65,start['y']+45,steps=6)
    page.mouse.up()
    assert svg.get_attribute('viewBox')!=initial
    # A drag's synthetic click must not prevent the next explicit toolbar action.
    page.locator('[data-graph-action="reset"]').click()
    assert svg.get_attribute('viewBox')==initial
    group=page.locator('.graph-node.evidence').first
    group.focus();group.press('Enter')
    expect(page.locator('.graph-expand')).to_be_visible()
    expect(group).to_have_attribute('aria-pressed','true')
    expect(page.locator('.graph-edge.is-active').first).to_be_visible()
    group.press(' ')
    expect(page.locator('.graph-expand')).to_be_visible()
    record('graph_zoom_pan_reset_and_keyboard')

    with page.expect_download() as info:
        page.get_by_role('button',name='下载完整图谱').click()
    path=RUN/'downloads/graph.json';info.value.save_as(str(path))
    exported=json.loads(path.read_text())
    assert exported==graphs[-1]
    assert len(exported['nodes'])>page.locator('.graph-node').count()
    record('graph_complete_original_json_export',nodes=len(exported['nodes']),edges=len(exported['edges']))

    page.set_viewport_size({'width':390,'height':844})
    expect(page.locator('.graph-explorer')).to_be_visible()
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(RUN/'graph-mobile.png'),full_page=True)
    record('graph_mobile_layout_no_horizontal_overflow')
    page.set_viewport_size({'width':1440,'height':1050})

    # An empty API result is simulated only in this browser; no database changes.
    url=BASE+'/api/graph/compound/'+samples[-1]['compound_id']
    page.route(url,lambda route:route.fulfill(json={'nodes':[],'edges':[],'counts':{}}))
    page.locator('#graph-form button').click()
    expect(page.locator('.graph-node')).to_have_count(0)
    expect(page.locator('.graph-inspector')).to_contain_text('暂无关联节点')
    page.locator('[data-graph-action="evidence"]').click()
    expect(page.locator('.graph-page-status')).to_have_text('暂无证据')
    expect(page.locator('[data-graph-action="next"]')).to_be_disabled()
    expect(page.locator('[data-graph-action="prev"]')).to_be_disabled()
    page.unroute(url)
    page.locator('#graph-form button').click()
    expect(page.locator('.graph-node').first).to_be_visible()
    record('graph_empty_state_and_recovery')
    page.screenshot(path=str(RUN/'graph-desktop.png'),full_page=True)

def check_completed_research_api(client):
    """Verify preserved research data independently of the workspace pages."""
    response=client.get('/api/workspace/research/summary')
    response.raise_for_status()
    summary=response.json()
    response=client.get('/api/workspace/research/endpoints')
    response.raise_for_status()
    endpoints=response.json()['items']
    assert summary['status']=='complete' and summary['quality_rows'] and endpoints
    selected=endpoints[0]
    response=client.get('/api/workspace/research/endpoints',
                        params={'method_id':selected['method_id'],'family':selected['family']})
    response.raise_for_status()
    wanted=[r for r in endpoints if r['method_id']==selected['method_id'] and r['family']==selected['family']]
    assert response.json()['items']==wanted
    record('api_research_quality_and_endpoint_filters',methods=len(summary['quality_rows']),endpoint_rows=len(endpoints))
    for case in summary['curve_cases']:
        response=client.get('/api/workspace/research/curves/'+case['compound'])
        response.raise_for_status()
        data=response.json()
        assert data['compound']==case['compound']
        assert len(data['curves'])==case['candidate_count']
        assert len(data['observation']['time_h'])==case['observation_count']
        assert len(data['observation']['time_h'])==len(data['observation']['concentration_ng_ml'])
        assert all(len(curve['time_h'])==len(curve['concentration_ng_ml']) for curve in data['curves'])
    record('api_real_prediction_observation_curves',cases=len(summary['curve_cases']))
    for item in summary['downloads']:
        response=client.get(item['url'])
        assert response.status_code==200 and len(response.content)>100
        if item['key']=='paper':assert response.content.startswith(b'%PDF-')
    record('api_research_result_downloads',files=len(summary['downloads']))


def check_blueprint_workspace(page):
    catalog=page.request.get(BASE+'/api/workspace/blueprint-cases').json()
    assert catalog['total']==len(catalog['items'])>0
    page.locator('nav a[href="#blueprints"]').click()
    expect(page.locator('#bp-workspace')).to_be_visible()
    expect(page.locator('[data-bp-input="case"] option')).to_have_count(catalog['total']+1)
    case=next(r for r in catalog['items'] if r['drug']=='Midazolam')
    page.locator('[data-bp-input="case"]').select_option(case['case_id'])
    expect(page.locator('.bp-case-name')).to_have_text(case['drug'])
    detail=page.request.get(BASE+'/api/workspace/blueprint-cases/'+case['case_id']).json()
    assert detail['curve']['time_h'][-1]==detail['request']['target']['window_end_h']
    page.locator('[data-bp-action="tab"][data-tab="structure"]').click()
    expect(page.locator('.bp-organ')).to_have_count(detail['structure']['organ_count'])
    page.locator('[data-bp-action="tab"][data-tab="parameters"]').click()
    expect(page.locator('#bp-detail-panel tbody tr')).to_have_count(len(detail['parameters']))
    page.locator('[data-bp-action="tab"][data-tab="evidence"]').click()
    expect(page.locator('#bp-detail-panel tbody tr')).to_have_count(len(detail['evidence']))
    expect(page.locator('[data-bp-region="result"] svg')).to_be_visible()
    assert all('/rest/pug/' not in (r.get('source_url') or '') for r in detail['evidence'])
    response=page.request.get(urljoin(BASE,detail['download_url'])+'&format=csv')
    assert response.status==200 and response.text().lstrip('\ufeff').startswith('time_h,concentration_ng_ml')
    record('browser_blueprint_cases_structure_parameters_evidence_and_curve',cases=catalog['total'])


if RESEARCH_API_ONLY:
    with httpx.Client(base_url=BASE,trust_env=False,timeout=120) as client:
        assert client.post('/auth/login',json={'access_code':KEY}).status_code==200
        check_completed_research_api(client)
    print(json.dumps({'passed':len(results),'url':BASE}),flush=True)
    raise SystemExit(0)

if not FOCUSED:
    with httpx.Client(base_url=BASE,trust_env=False,timeout=120) as c:
        assert c.get('/api/workspace/summary').status_code==401
        assert c.get('/api/compounds/search').status_code==401
        assert c.post('/auth/login',json={'access_code':'invalid'}).status_code==401
        assert c.post('/auth/login',json={'access_code':KEY},headers={'Origin':'http://untrusted.invalid'}).status_code==403
        assert c.post('/auth/login',json={'access_code':KEY}).status_code==200
        assert c.get('/api/workspace/summary',headers={'Host':'untrusted.invalid'}).status_code==400
        record('private_authentication_and_origin_guard')
        summary=c.get('/api/workspace/summary').json()['counts']
        assert all(summary[k]>0 for k in ('versions','evidence','covered_compounds','families'))
        assert summary['covered_compounds']<=summary['compounds'] and summary['families']<=summary['versions']
        record('expanded_live_database',**summary)
        versions=[]
        for offset in range(0,summary['versions'],24):
            rows=c.get('/api/workspace/models',params={'limit':24,'offset':offset}).json()
            versions.extend(r['model_version_id'] for r in rows['items'])
        assert len(versions)==summary['versions']==len(set(versions))
        record('complete_model_pagination',versions=len(versions))
        evidence=[]
        for offset in range(0,summary['evidence'],500):
            r=c.get('/api/evidence/search',params={'limit':500,'offset':offset});r.raise_for_status()
            evidence.extend((e['evidence_id'],e['compound_id']) for e in r.json())
        assert len(evidence)==summary['evidence']==len(set(evidence))
        record('complete_evidence_pagination',evidence=len(evidence))
        new=c.get('/api/evidence/search',params={'compound':'voriconazole','limit':10}).json()
        assert new
        e=new[0]; detail=c.get(f"/api/evidence/{e['evidence_type']}/{e['evidence_id']}");detail.raise_for_status()
        assert detail.json()['summary']['source_location']
        record('expanded_compound_evidence_and_provenance',compound='voriconazole')
        missing=c.get('/api/workspace/models/00000000-0000-0000-0000-000000000000')
        assert missing.status_code==404
        assert c.get('/api/workspace/models/not-a-uuid').status_code==422
        record('invalid_model_ids')
        target='/api/workspace/evidence/special_population/dc11b8e2-38ef-5f5e-be49-b8a277049293'
        provenance=c.get(target).json()['provenance']
        assert provenance['pmid']=='10197295'
        assert any(x['url']=='https://pubmed.ncbi.nlm.nih.gov/10197295/' for x in provenance['links'])
        assert not any('efetch' in x['url'] for x in provenance['links'])
        assert provenance['pdf_note']
        record('exact_article_provenance_not_batch_url')
        pdf_record=c.get('/api/workspace/evidence/clinical_pk/0d74dac3-7624-5909-ba49-a50c7eed7f77').json()
        pdf_url=next(x['url'] for x in pdf_record['provenance']['links'] if x['kind']=='pdf_local')
        pdf=c.get(pdf_url)
        assert pdf.status_code==200 and pdf.content.startswith(b'%PDF-')
        assert pdf.headers['content-type']=='application/pdf'
        assert pdf.headers['content-disposition'].startswith('inline;')
        assert httpx.get(BASE+pdf_url,trust_env=False).status_code==401
        record('authenticated_inline_pdf',bytes=len(pdf.content))
        check_completed_research_api(c)

with sync_playwright() as w:
    browser=w.chromium.launch(headless=True,args=['--no-sandbox'],downloads_path=str(RUN/'downloads'))
    page=browser.new_page(viewport={'width':1440,'height':1050},accept_downloads=True)
    page.set_default_timeout(60000)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(BASE)
    expect(page.locator('#login')).to_be_visible()
    page.get_by_label('个人访问码',exact=True).fill(KEY)
    page.get_by_role('button',name='进入工作台').click()
    expect(page.locator('#main h1')).to_have_text('从证据到模型，再到预测。')
    expect(page.locator('.metric')).to_have_count(4)
    page.evaluate('document.fonts.ready')
    page.screenshot(path=str(RUN/'overview.png'),full_page=True)
    record('browser_login_and_overview')
    record('browser_page_overview')
    navigation=page.locator('nav a')
    expect(navigation).to_have_count(6)
    assert navigation.evaluate_all('(links)=>links.map(link=>link.getAttribute("href"))')==[
        '#overview','#compounds','#models','#recommend','#blueprints','#graph']
    if FOCUSED:
        (check_graph if GRAPH_ONLY else check_saved_components if COMPONENTS_ONLY else check_blueprint_workspace)(page)
        assert not errors,errors
        record('browser_no_javascript_errors')
        browser.close()
        print(json.dumps({'passed':len(results),'url':BASE}),flush=True)
        raise SystemExit(0)
    for route,title in [('compounds','化合物与证据'),('models','公共模型库'),('recommend','推荐工作台'),('blueprints','模型蓝图与预测'),('graph','知识图谱')]:
        page.locator(f'nav a[href="#{route}"]').click()
        expect(page.locator('#main h1')).to_have_text(title)
        expect(page.locator('#main .error:visible, #main .bp-error:visible')).to_have_count(0)
        record('browser_page_'+route)
    page.locator('nav a[href="#compounds"]').click()
    page.get_by_role('button',name='证据检索',exact=True).click()
    page.locator('#evidence-form select[name="compound"]').select_option('Voriconazole')
    page.get_by_role('button',name='检索证据',exact=True).click()
    expect(page.locator('tbody tr').first).to_contain_text(re.compile('voriconazole',re.I))
    row=page.locator('tbody tr').first
    expect(row.locator('a[data-action="evidence-source"]')).to_have_count(1)
    expect(page.locator('[data-action="evidence-detail"]')).to_have_count(0)
    source_location=check_source_link(page, row.locator('a[data-action="evidence-source"]'))
    record('browser_new_evidence_search_and_source_link',source=source_location)
    with page.expect_download() as info:
        page.get_by_role('button',name='导出全部匹配证据').click()
    path=RUN/'downloads/evidence.csv';info.value.save_as(str(path))
    assert path.stat().st_size>100 and 'voriconazole' in path.read_text().lower()
    record('browser_filtered_evidence_csv_download')
    page.locator('#evidence-form select[name="compound"]').select_option('caffeine')
    page.locator('#evidence-form select[name="type"]').select_option('special_population')
    page.locator('#evidence-form input[name="q"]').fill('liver disease')
    page.get_by_role('button',name='检索证据',exact=True).click()
    link=page.locator('a[data-action="evidence-source"][data-id="dc11b8e2-38ef-5f5e-be49-b8a277049293"]')
    expect(link).to_have_count(1)
    check_source_link(page, link, 'https://pubmed.ncbi.nlm.nih.gov/10197295/')
    for technical_label in ('证据 ID','原始位置','文件校验值'):
        expect(page.locator('#main')).not_to_contain_text(technical_label)
    page.locator('#toast').evaluate('(e)=>e.hidden=true')
    page.screenshot(path=str(RUN/'evidence-provenance.png'),full_page=True)
    record('browser_single_article_link_without_technical_details')
    page.locator('nav a[href="#models"]').click()
    page.get_by_role('button',name='下一页 →').click()
    expect(page.locator('.pagination')).to_contain_text('第 2 页')
    page.get_by_role('button',name='详情 ↗').first.click()
    expect(page.locator('#detail')).to_be_visible()
    expect(page.locator('#detail-body')).to_contain_text('版本')
    page.get_by_role('button',name='关闭详情').click()
    with page.expect_download() as info:
        page.get_by_role('button',name='导出模型目录').click()
    path=RUN/'downloads/models.json';info.value.save_as(str(path))
    assert len(json.loads(path.read_text())['items'])==summary['versions']
    page.locator('#toast').evaluate('(e)=>e.hidden=true')
    page.screenshot(path=str(RUN/'models.png'),full_page=True)
    record('browser_model_detail_pagination_and_full_export')
    page.locator('nav a[href="#recommend"]').click()
    current=page.request.get(BASE+'/api/workspace/research/jobs/current').json()
    if current.get('status')=='complete':
        page.evaluate('(id)=>sessionStorage.setItem("pbpk_model_job",id)',current['job_id'])
        page.reload()
        expect(page.locator('.result-candidate')).to_have_count(len(current['result']['top_candidates']))
        expect(page.locator('#research-result')).to_contain_text('PBPK-Match')
        record('browser_saved_learning_recommendation_without_rerunning_gpu',candidates=current['result']['candidate_count'])
    page.locator('#research-recommend-form select[name="compound"]').select_option('midazolam')
    page.get_by_role('button',name='仅检查用途与适用条件',exact=True).click()
    expect(page.get_by_role('heading',name='模型匹配结果',exact=True)).to_be_visible()
    expect(page.locator('#main')).to_contain_text(f'扫描 {summary["versions"]:,} 个版本')
    assert page.locator('tbody tr').count()>0
    assert '[object Object]' not in page.locator('#main').inner_text()
    page.locator('#toast').evaluate('(e)=>e.hidden=true')
    page.screenshot(path=str(RUN/'recommendation.png'),full_page=True)
    record('browser_live_public_model_applicability_check',catalog_scanned=summary['versions'])
    check_saved_components(page)
    check_blueprint_workspace(page)
    check_graph(page)
    page.set_viewport_size({'width':390,'height':844})
    page.locator('nav a[href="#overview"]').click()
    expect(page.locator('.metric')).to_have_count(4)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(RUN/'mobile.png'),full_page=True)
    record('mobile_layout_no_horizontal_overflow')
    assert not errors,errors
    record('browser_no_javascript_errors')
    browser.close()
print(json.dumps({'passed':len(results),'url':BASE}),flush=True)
