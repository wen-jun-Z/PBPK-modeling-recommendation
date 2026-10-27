"""Real Chromium / WebSocket / browser-download checks against this project only."""
from pathlib import Path
import json,os,sys,time,re,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from service_runner import ROOT,DEPLOY,environment
os.environ.clear();os.environ.update(environment())
os.environ['LD_LIBRARY_PATH']=str(DEPLOY/'cache/browser-libs/extracted/usr/lib/x86_64-linux-gnu')
from playwright.sync_api import sync_playwright,expect
expect.set_options(timeout=60000)
ports=json.loads((DEPLOY/'ports.json').read_text())
access=json.loads((DEPLOY/'access.json').read_text())
base=f"http://127.0.0.1:{ports['ui']}"
results=[];out=DEPLOY/'browser';out.mkdir(exist_ok=True)
def record(name,details):
    results.append({'test':name,'status':'passed','details':details})
    (DEPLOY/'browser_test.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print(name,details,flush=True)
with sync_playwright() as w:
    browser=w.chromium.launch(headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
    page.set_default_timeout(60000)
    page.goto(base);page.locator('h1').wait_for()
    links=page.locator('[data-testid="stSidebarNav"] a').evaluate_all('(es)=>es.map(e=>({name:e.textContent,href:e.href}))')
    assert len(links)==14
    for item in links:
        page.goto(item['href']);page.locator('h1').wait_for()
        # Tables or explanatory page content must finish rendering without a Python exception.
        page.wait_for_function("document.querySelectorAll('[data-testid=stDataFrame], [data-testid=stMarkdownContainer]').length > 0")
        expect(page.locator('[data-testid="stException"]')).to_have_count(0)
        record('browser_page:'+item['name'],{'title':page.locator('h1').inner_text()})
    page.goto(base+'/evidence_search')
    page.get_by_label('Parameter or text',exact=True).fill('clearance')
    page.get_by_label('Parameter or text',exact=True).press('Enter')
    page.get_by_text('Hybrid',exact=True).click()
    page.get_by_role('heading',name='Vector Chunk Search',exact=True).wait_for()
    expect(page.locator('[data-testid="stDataFrame"]')).to_have_count(2)
    record('browser_hybrid_search',{'structured_and_semantic_tables':2})
    page.screenshot(path=str(out/'knowledge_search.png'),full_page=True)
    page.goto(base+'/exports')
    page.get_by_role('button',name='Export evidence CSV',exact=True).click()
    with page.expect_download() as event:
        page.get_by_role('button',name='Download export',exact=True).click()
    download=event.value;target=out/'evidence.csv';download.save_as(str(target))
    assert target.stat().st_size>100 and 'midazolam' in target.read_text().lower()
    record('browser_evidence_download',{'bytes':target.stat().st_size})
    page.goto(base+'/knowledge_graph')
    page.get_by_role('tab',name='Export JSON',exact=True).click()
    with page.expect_download() as event:
        page.get_by_role('button',name='Download graph JSON',exact=True).click()
    target=out/'graph.json';event.value.save_as(str(target));graph=json.loads(target.read_text())
    assert graph['nodes'] and graph['edges']
    record('browser_graph_download',{'nodes':len(graph['nodes']),'edges':len(graph['edges'])})
    page.goto(base+'/recommendation_agent')
    page.get_by_role('button',name='Rank downloaded public models',exact=True).click()
    page.get_by_text('Recommended use',exact=True).wait_for()
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)
    page.screenshot(path=str(out/'recommendation.png'),full_page=True)
    record('browser_model_ranking_submit',{'result_rendered':True})
    page.get_by_role('combobox',name='Target compound',exact=True).click()
    page.get_by_role('option',name=re.compile('^caffeine$',re.I)).click()
    expect(page.get_by_role('textbox',name='Dose unit',exact=True)).to_have_value('mg/kg')
    page.wait_for_function('document.querySelector("[data-testid=stApp]").getAttribute("data-test-script-state") === "notRunning"')
    page.get_by_role('button',name='Rank downloaded public models',exact=True).click()
    page.get_by_text('No same-compound model',exact=True).wait_for()
    page.wait_for_function('document.querySelector("[data-testid=stApp]").getAttribute("data-test-script-state") === "notRunning"')
    page.get_by_role('button',name='Check registered executable blueprint',exact=True).click()
    page.get_by_role('heading',name='Executable blueprint workspace',exact=True).wait_for()
    with page.expect_download() as event:
        page.get_by_role('button',name='Download decision package',exact=True).click()
    target=out/'blueprint.zip';event.value.save_as(str(target))
    assert zipfile.is_zipfile(target)
    record('browser_blueprint_zip_download',{'bytes':target.stat().st_size})
    browser.close()
print(json.dumps({'passed':len(results),'base_url':base}),flush=True)
