"""Real bounded workspace prediction and its integrity/access contracts."""
from copy import deepcopy
import hashlib
import io
import json
import runpy
from pathlib import Path
import time
import zipfile

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
from pydantic import ValidationError

from pbpk_kb.web import blueprints as web
from pbpk_kb.web import blueprint_human as human_builder


@pytest.fixture(scope='module')
def bundle():
    case = next(row for row in web.cases()['items'] if row['drug'] == 'Midazolam')
    return web._bundle(case['case_id'])


def body(bundle, **changes):
    req=bundle['request']
    return web.PredictionRequest(case_id=bundle['case']['case_id'],
        **({'dose_mg':req['dosing']['dose_mg'],
            'infusion_minutes':req['dosing']['infusion_minutes'],
            'window_end_h':req['target']['window_end_h']} | changes))


def test_real_prediction_conserves_mass_and_responds_to_new_dose(bundle):
    digest=hashlib.sha256(json.dumps(bundle,sort_keys=True).encode()).hexdigest()
    one,package=web._run_prediction(body(bundle),bundle,lambda *_:None)
    two,_=web._run_prediction(body(bundle,dose_mg=2*bundle['request']['dosing']['dose_mg']),bundle,lambda *_:None)
    assert one['mass_balance']['maximum_relative_error'] < 1e-6
    np.testing.assert_allclose(np.array(two['curve']['concentration_ng_ml']),
                               2*np.array(one['curve']['concentration_ng_ml']),rtol=1e-8)
    assert two['scenario_changed'] is True
    assert one['scientific_accepted'] is False
    assert len(package['curve']['time_min']) > 700
    assert one['structure']['organ_count']==15
    assert hashlib.sha256(json.dumps(bundle,sort_keys=True).encode()).hexdigest()==digest


def test_real_custom_development_curve_calibration(bundle):
    req=bundle['request'];times=np.array([.3,.5,.8,1.,2.,3.,4.,6.,8.])
    expected=web.model.simulate(bundle['physiology'],req,times,bundle['fitted']['parameters']['log_parameters'])
    # Synthetic observations exercise recovery, and are explicitly labelled as such.
    dev={'source_label':'Synthetic unit test from fixed model, not clinical evidence',
         'human':human_input(bundle),
         'dose_mg':req['dosing']['dose_mg'],'infusion_minutes':req['dosing']['infusion_minutes'],
         'observations':[{'time_h':float(t),'concentration_ng_ml':float(c)*1000}
                         for t,c in zip(times,expected['concentration_mg_l'])]}
    result,package=web._run_prediction(body(bundle,development_curve=dev),bundle,lambda *_:None)
    assert result['calibration']['mode']=='user_development'
    assert 0<result['calibration']['calls']<=4096
    assert result['calibration']['target_observations_used'] is False
    assert result['calibration']['development_gmfe']<1.3
    assert result['calibration']['human']==req['human']
    assert result['calibration']['dosing']==req['dosing']
    assert package['input']['development_curve']['source_label'].startswith('Synthetic')
    assert package['input']['development_curve']['human']==human_input(bundle)


def test_complete_catalog_and_case_source_values(bundle):
    rows=web.cases()['items']
    assert len(rows)==30 and len({r['case_id'] for r in rows})==30
    detail=web.case_detail(bundle['case']['case_id'])
    assert detail['request']['human']==bundle['request']['human']
    assert detail['curve']['time_h'][-1]==bundle['request']['target']['window_end_h']
    assert detail['evidence'][0]['value']==bundle['request']['evidence'][0]['value']
    assert all(link['url'].startswith(('http://','https://','/api/')) for link in detail['sources'])
    assert web._canonical_source_url('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/4192/property/MolecularWeight/JSON',bundle)=='https://pubchem.ncbi.nlm.nih.gov/compound/4192'
    assert not any('/rest/pug/' in row['source_url'] for row in detail['evidence'] if row['source_url'])


def test_saved_calibration_discloses_actual_development_conditions_not_prediction_human():
    case=next(row for row in web.cases()['items'] if row['drug']=='Caffeine')
    bundle=web._bundle(case['case_id'])
    detail=web.case_detail(case['case_id'])
    calibration=detail['calibration']
    assert calibration['human']==bundle['training']['human']
    assert calibration['dosing']==bundle['training']['dosing']
    assert calibration['study_id']==bundle['training']['study_id']
    assert calibration['source_label']
    assert calibration['visible_points']==len(bundle['training']['observations'])==5
    assert calibration['human']['age_years']==21.8
    assert calibration['human']['weight_kg']==70.
    assert calibration['dosing']['dose_mg']==350.
    assert calibration['dosing']['infusion_minutes']==30.
    assert detail['request']['human']['age_years']==26.
    assert detail['request']['human']['weight_kg']==74.
    assert detail['human_context']['reference_human']==bundle['request']['human']
    assert detail['human_context']['reference_kind']=='case_target'
    assert calibration['human']!=detail['request']['human']
    assert calibration['dosing']!=detail['request']['dosing']


@pytest.mark.parametrize('drug,parameter,expected', [
    ('Caffeine','molecular_weight','https://pubchem.ncbi.nlm.nih.gov/compound/2519'),
    ('Caffeine','solubility','https://pubchem.ncbi.nlm.nih.gov/compound/2519'),
    ('Alfentanil','binding_partner','https://www.medicines.org.uk/emc/product/6430/smpc'),
    ('Alfentanil','fraction_unbound','https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=bb56df52-5abf-47dc-ab8d-902cfe19ccb2'),
    ('Fluconazole','total_clearance','https://pubmed.ncbi.nlm.nih.gov/8444063/'),
    ('Fluconazole','fraction_unbound','https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=2a3ba181-5baa-5942-e063-6294a90a82b2'),
    ('Furosemide','total_clearance','https://pubmed.ncbi.nlm.nih.gov/6107232/'),
    ('Dexmedetomidine','total_clearance','https://www.accessdata.fda.gov/drugsatfda_docs/label/1999/21038lbl.pdf'),
    ('Sildenafil','pka_types','https://pmc.ncbi.nlm.nih.gov/articles/PMC10746450/'),
    ('Ethinylestradiol','pka_types','https://www.rsc.org/suppdata/d3/ay/d3ay01201h/d3ay01201h1.pdf'),
    ('Moclobemide','pka_types','https://pdf.hres.ca/dpd_pm/00035151.PDF'),
    ('Midazolam','total_clearance',None),
    ('Midazolam','solubility',None),
])
def test_blueprint_evidence_source_matches_drug_parameter_and_document(drug,parameter,expected):
    case=next(row for row in web.cases()['items'] if row['drug']==drug)
    b=web._bundle(case['case_id'])
    row=next(row for row in b['request']['evidence'] if row['parameter']==parameter)
    assert web._source_link(b,row)==expected


def test_ambiguous_source_and_cited_articles_do_not_supply_a_link():
    source={'source':{'location':'resultList.result'},'parameter':'total_clearance'}
    content=json.dumps({'resultList':{'result':[{'source':'MED','id':'11'},{'source':'MED','id':'22'}]}}).encode()
    assert web.blueprint_evidence_url(content,source,'example','pubchem:1') is None
    source['source']['location']='resultList.result[id=22].abstractText'
    assert web.blueprint_evidence_url(content,source,'example','pubchem:1')=='https://pubmed.ncbi.nlm.nih.gov/22/'
    cited=b'<article><back><ref-list><ref><article-id pub-id-type="pmid">11</article-id></ref></ref-list></back></article>'
    assert web.blueprint_evidence_url(cited,source,'example','pubchem:1') is None


def test_request_rejects_unsupported_and_invalid_inputs(bundle):
    with pytest.raises(ValidationError):body(bundle,route='oral')
    with pytest.raises(ValidationError):body(bundle,dose_mg=float('nan'))
    with pytest.raises(ValidationError):body(bundle,window_end_h=.1,infusion_minutes=10)
    with pytest.raises(ValidationError):
        web.DevelopmentCurve(source_label='invalid',source_url='javascript:alert(1)',dose_mg=1.,
            human=human_input(bundle),infusion_minutes=0.,
            observations=[{'time_h':float(i),'concentration_ng_ml':1.} for i in range(5)])


def human_input(bundle, **changes):
    return {key:bundle['request']['human'][key]
            for key in ('age_years','sex','weight_kg','height_cm')} | changes


def development_input(bundle, human=None):
    return {'source_label':'Synthetic routing fixture, not clinical evidence',
            'human':human if human is not None else human_input(bundle),
            'dose_mg':2*bundle['request']['dosing']['dose_mg'],'infusion_minutes':0.,
            'observations':[{'time_h':t,'concentration_ng_ml':10./(1+t)}
                            for t in (.1,.5,1.,2.,4.)]}


@pytest.mark.parametrize('human', [None, {'weight_kg':80.}])
def test_development_curve_requires_its_own_complete_human_metadata(bundle, human):
    dev=development_input(bundle)
    if human is None:
        dev.pop('human')
    else:
        dev['human']=human
    # A target profile must never silently become missing observation metadata.
    with pytest.raises(ValidationError):
        body(bundle,human=human_input(bundle),development_curve=dev)


def test_development_and_prediction_humans_must_match(bundle):
    with pytest.raises(ValidationError):
        body(bundle,human=human_input(bundle,weight_kg=80.),
             development_curve=development_input(bundle))


@pytest.mark.parametrize('change', [
    {'age_years':17.9}, {'age_years':80.1}, {'age_years':float('nan')},
    {'weight_kg':0}, {'weight_kg':300.1}, {'weight_kg':float('inf')},
    {'height_cm':0}, {'height_cm':250.1}, {'height_cm':float('nan')},
    {'sex':'unknown'}, {'population':'unsupported'},
])
def test_custom_human_rejects_invalid_or_unexposed_conditions(bundle, change):
    with pytest.raises(ValidationError):
        body(bundle,human=human_input(bundle,**change))


def test_custom_human_requires_a_complete_profile(bundle):
    with pytest.raises(ValidationError):
        body(bundle,human={'weight_kg':80})


@pytest.mark.parametrize('explicit', [False, True])
def test_unchanged_human_reuses_the_case_physiology(bundle, monkeypatch, explicit):
    def unexpected_rebuild(*args, **kwargs):
        pytest.fail('An unchanged human must not launch a native rebuild')
    monkeypatch.setattr(web,'rebuild_physiology',unexpected_rebuild)
    request=body(bundle,**({'human':human_input(bundle)} if explicit else {}))
    result,package=web._run_prediction(request,bundle,lambda *_:None)
    expected={'changed':False,'reference_human':bundle['request']['human'],
              'active_human':bundle['request']['human'],'physiology_source':'case',
              'calibration_mode':'case_saved','reference_kind':'case_target'}
    for key,value in expected.items():
        assert result['human_context'][key]==value
        assert package['human_context'][key]==value
    assert package['physiology']==bundle['physiology']


def test_custom_human_without_observations_is_rejected_before_build_or_submission(bundle,tmp_path,monkeypatch):
    run=tmp_path/'not-created'
    monkeypatch.setattr(web,'RUN',run)
    def unexpected_work(*args,**kwargs):
        pytest.fail('A custom human without observations must not launch work')
    monkeypatch.setattr(web,'rebuild_physiology',unexpected_work)
    monkeypatch.setattr(web.model,'simulate',unexpected_work)
    monkeypatch.setattr(web.fitter,'fit',unexpected_work)
    monkeypatch.setattr(web._pool,'submit',unexpected_work)
    request=body(bundle,human=human_input(bundle,weight_kg=80.))
    with pytest.raises(ValueError):
        web._run_prediction(request,bundle,lambda *_:None,output=tmp_path/'prediction')
    app=FastAPI();app.include_router(web.router)
    with TestClient(app) as client:
        # Repeated invalid requests must neither consume slots nor create jobs.
        for _ in range(3):
            response=client.post('/workspace/blueprint-predictions',json=request.model_dump())
            assert response.status_code==422,response.json()
        assert client.get('/workspace/blueprint-predictions').json()['total']==0
    assert not run.exists()
    assert not (tmp_path/'prediction').exists()


@pytest.mark.parametrize('explicit_human', [False, True])
def test_custom_human_drives_prediction_fit_presentation_and_download(bundle, tmp_path, monkeypatch, explicit_human):
    # The real native physiology and solver are covered by deployment E2E.
    # This routing fixture catches reuse of frozen anatomy anywhere in the UI path.
    bundle=deepcopy(bundle)
    # A request may predate formula selection; all new stages must retain the
    # calibrated formula, including when user observations trigger a refit.
    bundle['request']['partition_method']=('PKSimStandard' if bundle['fitted']['formula']!='PKSimStandard'
                                           else 'RodgersRowland')
    frozen=deepcopy(bundle)
    changed=human_input(bundle,age_years=46.,sex='FEMALE',weight_kg=60.,height_cm=162.)
    active=bundle['request']['human'] | changed
    anatomy=deepcopy(bundle['physiology'])
    anatomy['organs'][0]['volume_l']*=1.15
    anatomy['organs'][0]['flow_l_h']*=1.05
    rebuilt=[];simulated=[];fitted=[]

    def rebuild(request, output, root, *, before_trial):
        before_trial()
        assert request['human']==active
        assert request['partition_method']==bundle['fitted']['formula']
        assert root==web.ROOT
        assert Path(output).is_relative_to(tmp_path)
        rebuilt.append(deepcopy(request))
        return anatomy

    def simulate(physiology, request, times, parameters):
        assert physiology==anatomy
        assert request['human']==active
        assert request['partition_method']==bundle['fitted']['formula']
        simulated.append(deepcopy(request))
        return {'concentration_mg_l':anatomy['organs'][0]['volume_l']*np.exp(-np.asarray(times)),
                'observation':'synthetic routing fixture',
                'mass_balance':{'fixture':True}}

    def fit(simulator, request, training, physiology, **kwargs):
        assert physiology==anatomy
        assert request['human']==training['human']==active
        assert request['partition_method']==bundle['fitted']['formula']
        assert request['dosing']['dose_mg']==2*bundle['request']['dosing']['dose_mg']
        assert training['role']=='user_declared_development'
        kwargs['before_trial']()
        simulator(bundle['fitted']['parameters']['log_parameters'],np.array([.1,.5,1.]))
        fitted.append(deepcopy(training))
        parameters=deepcopy(bundle['fitted']['parameters'])
        parameters['development_gmfe']=1.
        return {'best':parameters,'calls':1}

    monkeypatch.setattr(web,'rebuild_physiology',rebuild)
    monkeypatch.setattr(web.model,'simulate',simulate)
    monkeypatch.setattr(web.fitter,'fit',fit)
    extra={'development_curve':development_input(bundle,changed)}
    if explicit_human:
        extra['human']=changed
    result,package=web._run_prediction(body(bundle,**extra),bundle,
                                     lambda *_:None,output=tmp_path/'prediction')
    assert len(rebuilt)==1
    assert len(fitted)==1
    assert len(simulated)==2
    assert result['scenario_changed'] is True
    assert result['request']['human']==active
    assert result['request']['partition_method']==bundle['fitted']['formula']
    assert result['structure']['organs'][0]['volume_l']==anatomy['organs'][0]['volume_l']
    assert result['structure']['organs'][0]['flow_l_h']==anatomy['organs'][0]['flow_l_h']
    expected={'changed':True,'reference_human':bundle['request']['human'],
              'active_human':active,'physiology_source':'rebuilt',
              'calibration_mode':'development_refit','reference_kind':'case_target'}
    for key,value in expected.items():
        assert result['human_context'][key]==value
        assert package['human_context'][key]==value
    assert result['scientific_accepted'] is package['scientific_accepted'] is False
    assert package['physiology']==anatomy
    assert package['request']['human']==active
    assert result['calibration']['human']==active
    assert result['calibration']['dosing']==fitted[0]['dosing']
    assert result['calibration']['dosing']['dose_mg']!=package['request']['dosing']['dose_mg']
    assert package['input']['development_curve']['human']==changed
    archive=web._download(package,'custom-human','zip')
    with zipfile.ZipFile(io.BytesIO(archive.body)) as zipped:
        exported=json.loads(zipped.read('blueprint.json'))
        assert exported['physiology']==anatomy
        assert exported['request']['human']==active
        assert exported['request']['partition_method']==bundle['fitted']['formula']
        assert exported['human_context']==package['human_context']
    assert bundle==frozen


def test_custom_human_rebuild_failure_never_uses_the_original_anatomy(bundle, tmp_path, monkeypatch):
    def failed_rebuild(*args, **kwargs):
        raise ValueError('Human construction failed: synthetic error')
    def unexpected_prediction(*args, **kwargs):
        pytest.fail('A failed human rebuild must not silently reuse the old anatomy')
    monkeypatch.setattr(web,'rebuild_physiology',failed_rebuild)
    monkeypatch.setattr(web.model,'simulate',unexpected_prediction)
    changed=human_input(bundle,weight_kg=80.)
    with pytest.raises(ValueError,match='Human construction failed'):
        web._run_prediction(body(bundle,human=changed,development_curve=development_input(bundle,changed)),
                            bundle,lambda *_:None,output=tmp_path/'prediction')


def test_native_human_builder_rejects_existing_output_without_writing(bundle, tmp_path, monkeypatch):
    output=tmp_path/'existing-human';output.mkdir()
    marker=output/'keep.txt';marker.write_text('existing result')
    def unexpected_build(*args, **kwargs):
        pytest.fail('Existing native output must not be overwritten or reused')
    monkeypatch.setattr(human_builder,'_physiology',unexpected_build)
    with pytest.raises(ValueError,match='new job directory'):
        human_builder.rebuild_physiology(bundle['request'],output,web.ROOT)
    assert marker.read_text()=='existing result'


@pytest.mark.parametrize('path', [
    'runtime/not-workspace-human-output',
    'runtime/deployment/workspace',
    'runtime/deployment/workspace/../not-workspace-human-output',
])
def test_native_human_builder_rejects_output_outside_its_workspace(bundle, monkeypatch, path):
    def unexpected_build(*args, **kwargs):
        pytest.fail('Out-of-scope paths must be rejected before native execution')
    monkeypatch.setattr(human_builder,'_physiology',unexpected_build)
    with pytest.raises(ValueError,match='workspace'):
        human_builder.rebuild_physiology(bundle['request'],web.ROOT/path,web.ROOT)


def test_native_human_builder_stops_before_execution_when_workspace_limit_is_exceeded(bundle, tmp_path, monkeypatch):
    output=tmp_path/'over-quota-human'
    monkeypatch.setattr(human_builder,'_native_workspace_bytes',
                        lambda *_:human_builder.WORKSPACE_NATIVE_LIMIT_BYTES+1)
    def unexpected_build(*args, **kwargs):
        pytest.fail('Workspace quota must prevent the next native build')
    monkeypatch.setattr(human_builder,'_physiology',unexpected_build)
    with pytest.raises(ValueError,match='2 GiB'):
        human_builder.rebuild_physiology(bundle['request'],output,web.ROOT)
    assert not output.exists()


def test_native_human_builder_reports_infrastructure_failure_without_fallback(bundle, tmp_path, monkeypatch):
    calls=[]
    failure=human_builder.InfrastructurePause('synthetic native validation failure')
    def failed_build(request, output, root, before_trial):
        before_trial()
        calls.append(request)
        raise failure
    monkeypatch.setattr(human_builder,'_native_workspace_bytes',lambda *_:0)
    monkeypatch.setattr(human_builder,'_physiology',failed_build)
    with pytest.raises(ValueError,match='Old physiology was not substituted') as error:
        human_builder.rebuild_physiology(bundle['request'],tmp_path/'failed-human',web.ROOT)
    assert error.value.__cause__ is failure
    assert len(calls)==1


def test_integrity_and_project_boundary(tmp_path):
    source=tmp_path/'source.json';source.write_text('{"value":1}')
    pin={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    assert web._ref(pin)=={'value':1}
    source.write_text('{"value":9}')
    with pytest.raises(ValueError,match='verification'):web._ref(pin)
    with pytest.raises(ValueError,match='project'):web._path('/outside/not-a-project-file')
    with pytest.raises(ValueError,match='project'):web._path(web.ROOT/'../not-a-project-file')


def test_async_history_download_and_restart(tmp_path,bundle,monkeypatch):
    monkeypatch.setattr(web,'RUN',tmp_path/'blueprints')
    app=FastAPI();app.include_router(web.router);client=TestClient(app)
    response=client.post('/workspace/blueprint-predictions',json=body(bundle).model_dump())
    assert response.status_code==202
    job_id=response.json()['job_id']
    for _ in range(200):
        job=client.get('/workspace/blueprint-predictions/'+job_id).json()
        if job['status'] not in ('queued','running'):break
        time.sleep(.05)
    assert job['status']=='completed',job
    assert client.get('/workspace/blueprint-predictions').json()['total']==1
    archive=client.get(f'/workspace/blueprint-predictions/{job_id}/download')
    assert archive.status_code==200
    with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
        assert {'blueprint.json','concentration.csv','replay.py'} <= set(z.namelist())
        assert json.loads(z.read('blueprint.json'))['request']['compound_id']==bundle['request']['compound_id']
        replay_dir=tmp_path/'replay';replay_dir.mkdir()
        for name in ('blueprint.json','replay.py'):
            (replay_dir/name).write_bytes(z.read(name))
        with monkeypatch.context() as scoped:
            scoped.chdir(replay_dir)
            runpy.run_path(str(replay_dir/'replay.py'))
        replayed=json.loads((replay_dir/'replayed_prediction.json').read_text())
        original=json.loads((replay_dir/'blueprint.json').read_text())
        mw=next(float(e['value']) for e in original['request']['evidence'] if e['parameter']=='molecular_weight')
        np.testing.assert_allclose(replayed['concentration_mg_l'],
            np.array(original['curve']['concentration'])*mw/1000,rtol=1e-10,atol=1e-14)
    response=client.get(f'/workspace/blueprint-predictions/{job_id}/download?format=csv')
    assert response.text.lstrip('\ufeff').startswith('time_h,concentration_ng_ml')
    path=web._job_path(job_id);doc=web._read(path)
    doc.update(status='running',service_instance='earlier-server');web._save(path,doc)
    assert client.get('/workspace/blueprint-predictions/'+job_id).json()['status']=='interrupted'
    assert client.get(f'/workspace/blueprint-predictions/{job_id}/download').status_code==409


def test_unknown_case_is_not_substituted():
    with pytest.raises(web.HTTPException) as error:web.case_detail('unknown_drug')
    assert error.value.status_code==404
