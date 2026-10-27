"""Synthetic external-prediction contracts; never read held-out PK outcomes."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('vori_methods_prediction', ROOT/'scripts/predict_voriconazole_methods_external.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def write(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


@pytest.fixture
def registered(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    specs = [{'parameter_id':'x','lower_bound':0.,'upper_bound':10.,'initial_value':1.},
             {'parameter_id':'y','lower_bound':0.,'upper_bound':10.,'initial_value':2.}]
    case = {'case_id':'synthetic_case','compound_id':'synthetic_compound','parameter_search':{'parameters':specs}}
    candidates=[]
    for action in sorted(p.PARTITION_ACTIONS):
        for bits,free in [('00',[]),('01',['y']),('10',['x']),('11',['x','y'])]:
            candidates.append({'candidate_id':action+'__'+bits,'action':action,'structure_id':action,
                'parameter_ids':['x','y'],'free_parameter_ids':free,
                'parameterization_hash':p.digest({'specs':specs,'free':free})})
    monkeypatch.setattr(p,'candidate_space',lambda case,root:candidates)
    package=write(tmp_path/'input.json',{'case':case});suite=tmp_path/'suite';code=tmp_path/'code.py';code.write_text('# synthetic\n')
    code_hash={'code.py':p.file_sha(code)}
    config={'version':'blueprint-reviewed-development-suite-v1','tasks':[],'code_sha256':code_hash}
    for block,(budget,count,methods) in p.BLOCKS.items():
        jobs=[{'method':method,'seed':seed} for method,seed in sorted(p.expected_jobs(methods))]
        task_id=block+'__synthetic_case';directory=suite/'tasks'/task_id
        config['tasks'].append({'case_id':case['case_id'],'task_id':task_id,'block_id':block,'budget':budget,
            'matched_count':count,'compound_id':case['compound_id'],'bundle_path':'input.json',
            'bundle_sha256':p.file_sha(package),'jobs':jobs})
        cp=write(directory/'case.json',case)
        tc={'case_sha256':p.file_sha(cp),'case_bundle_sha256':p.file_sha(package),'case_bundle_path':'input.json',
            'compound_id':case['compound_id'],'max_solver_calls_per_method':budget,'exact_free_parameter_count':count,
            'jobs':jobs,'external_observations_available_to_optimizer':False,'independent_test_loaded':False,
            'code_sha256':code_hash,'candidates':candidates}
        write(directory/'configuration.json',tc)
        rows=[{**j,'status':'blocked','prediction_available':False,'selected_rebuild_verified':False,
            'blocked_reason':'synthetic unavailable'} for j in jobs]
        write(directory/'results.json',{'completed_jobs':len(jobs),'methods':rows})
    write(suite/'configuration.json',config);write(suite/'completion.json',{'status':'finished','halt_reason':None})
    return tmp_path,suite,case,package,candidates


def available(registered):
    root,suite,case,package,candidates=registered
    directory=suite/'tasks'/'main_5000__synthetic_case'
    result=json.loads((directory/'results.json').read_text())
    row=next(x for x in result['methods'] if x['method']=='M0')
    candidate=next(c for c in candidates if c['candidate_id']=='partition_pksim_standard__11')
    row.update(status='completed',prediction_available=True,selected_rebuild_verified=True,
        selected_candidate_id=candidate['candidate_id'],effective_free_parameter_ids=['x','y'])
    write(directory/'results.json',result)
    parameter_path=write(directory/'methods/M0_seed_17/selected_parameters.json',{
        'parameter_ids':['x','y'],'calibration_source':case['case_id'],'scientific_accepted':False,'values':[1.5,2.5]})
    return parameter_path


def test_all_57_unavailable_are_retained(registered):
    _,suite,case,package,_=registered
    selections,models=p.load_selections(suite,case,package,{})
    assert len(selections)==57 and len({s['selection_id'] for s in selections})==57
    assert models=={} and all(s['model_key'] is None and s['failure'] for s in selections)


def test_valid_selection_bound_to_source_vector(registered):
    path=available(registered);_,suite,case,package,_=registered;frozen={}
    selections,models=p.load_selections(suite,case,package,frozen)
    assert len(selections)==57 and len(models)==1
    model=next(iter(models.values()))
    assert model['values']==[1.5,2.5] and model['parameters_sha256']==p.file_sha(path)
    assert str(path.relative_to(p.ROOT)) in frozen


@pytest.mark.parametrize('change',['missing_result','duplicate_job','different_case','wrong_code','candidate','invalid_bound','bool_parameter','nonestimated_change'])
def test_upstream_drift_or_denominator_loss_rejected(registered,change):
    parameter=available(registered);_,suite,case,package,_=registered
    directory=suite/'tasks/main_5000__synthetic_case';rp=directory/'results.json';result=json.loads(rp.read_text())
    cp=directory/'configuration.json';config=json.loads(cp.read_text())
    if change=='missing_result':result['methods'].pop();write(rp,result)
    elif change=='duplicate_job':result['methods'].append(result['methods'][0]);write(rp,result)
    elif change=='different_case':write(directory/'case.json',{**case,'compound_id':'foreign'})
    elif change=='wrong_code':(p.ROOT/'code.py').write_text('# tampered')
    elif change=='candidate':config['candidates'][0]['action']='foreign';write(cp,config)
    elif change=='invalid_bound':obj=json.loads(parameter.read_text());obj['values'][0]=100;write(parameter,obj)
    elif change=='bool_parameter':obj=json.loads(parameter.read_text());obj['values'][0]=True;write(parameter,obj)
    elif change=='nonestimated_change':
        next(x for x in result['methods'] if x['method']=='M0')['effective_free_parameter_ids']=[];write(rp,result)
    with pytest.raises((ValueError,KeyError)):
        p.load_selections(suite,case,package,{})


def test_original_nominal_sample_not_infusion_end():
    targets=[0.,1.33*60,80.,90.]
    assert p.original_grid_values([0.,79.,80.,90.],[0.,1.,2.,1.],targets)==pytest.approx([0.,1.8,2.,1.])


@pytest.mark.parametrize('times,values,targets',[
    ([0.,1.,2.],[0.,1.,-1e-15],[0.,1.]),
    ([0.,1.],[0.,float('nan')],[0.,1.]),
    ([0.,0.],[0.,1.],[0.]),
    ([0.,1.],[0.,1.],[0.,2.]),
])
def test_no_negative_tail_clipping_or_extrapolation(times,values,targets):
    with pytest.raises(ValueError):p.original_grid_values(times,values,targets)


def test_clinical_request_removes_historical_cotherapy(monkeypatch):
    monkeypatch.setattr(p,'ref',lambda *args:{'source':'metadata only'})
    monkeypatch.setattr(p,'BlueprintDesignRequest',SimpleNamespace(model_validate=lambda x:SimpleNamespace(model_dump=lambda **kw:x)))
    template={'studies':['old'],'requirement':{},'instance':{
        'individual':{'Name':'source','Seed':-17,'OriginData':{'Population':'source'}},
        'compound':{'Name':'Voriconazole1'},'expression_profiles':[{'source':'expression'}],
        'source_coadministration':{'compound':'Midazolam'},'protocol_representation':'clinical_repeated_dosing',
        'formulation':{'oral':True},'formulation_source':{'old':True}}}
    design={'population':{'age_mean_years':30,'weight_mean_kg':70,'height_mean_cm':175},
        'dose':{'value':4,'infusion_duration_min':80}}
    request=p.make_request(template,design,Path('unused'),'male')
    assert request['instance']['source_coadministration'] is None
    assert request['instance']['protocol_representation']=='clinical_regimen'
    assert request['instance']['formulation'] is None
    assert request['instance']['individual']['Seed']==-17
    assert request['instance']['compound']==template['instance']['compound']
    assert request['instance']['protocol']['Parameters'][1]['Value']==4
    assert request['instance']['protocol']['Parameters'][2]['Value']==80
    assert template['instance']['source_coadministration'] is not None


def test_lossless_native_model_archive(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    directory=tmp_path/'native';directory.mkdir();path=directory/'a.pkml'
    raw=b'<PKML GUID="random-not-reconstructable">\n'+b'abc'*10000+b'</PKML>'
    path.write_bytes(raw)
    manifest={'pkml_outputs':[{'path':'a.pkml','sha256':p.file_sha(path),'byte_size':len(raw)}]}
    archives=p.archive_pkml(directory,manifest,tmp_path)
    assert not path.exists() and len(archives)==1
    record=archives[0];zipped=tmp_path/record['path']
    assert gzip.decompress(zipped.read_bytes())==raw
    assert record['original_sha256']==hashlib.sha256(raw).hexdigest()
    assert record['sha256']==p.file_sha(zipped)


@pytest.mark.parametrize('resource', ['wall','native_failures','compile_failures'])
def test_resource_exhaustion_preserves_all_registered_predictions(tmp_path,monkeypatch,resource):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'STAGE',tmp_path/'stage')
    designs={d:{'case_id':f'case{d}'} for d in (4,6)}
    paths={d:write(tmp_path/f'design{d}.json',designs[d]) for d in designs}
    selections=[{'selection_id':str(i),'model_key':'a' if i%2 else 'b'} for i in range(57)]
    models={'a':{'action':'partition_pksim_standard','parameters_path':'params.json','values':[1.]},
            'b':{'action':'partition_rodgers_rowland','parameters_path':'params.json','values':[1.]}}
    write(tmp_path/'params.json',{'values':[1.]})
    monkeypatch.setattr(p,'protocol_record',lambda d,path,t,s,m,e:{'case_id':d['case_id'],'models':m,'selections':s,'sampling_time_min':[0.,60.],'evaluation_contract':e})
    monkeypatch.setattr(p,'load_scoring_contract',lambda frozen:{'version':'synthetic'})
    monkeypatch.setattr(p,'make_request',lambda *args:{'synthetic':True})
    monkeypatch.setattr(p,'audit_design_inputs',lambda *args,**kw:None)
    calls=[]
    def fail_native(**kw):
        calls.append(kw)
        raise RuntimeError('synthetic native failure')
    monkeypatch.setattr(p,'ref',lambda *args:{'source':'synthetic parameters'})
    def fail_compile(*args,**kw):
        raise RuntimeError('synthetic candidate compile failure before native invocation')
    monkeypatch.setattr(p,'compile_native_candidate',fail_compile if resource=='compile_failures' else lambda *args,**kw:{'snapshot':{}})
    monkeypatch.setattr(p,'execute_snapshot',fail_native if resource=='native_failures' else lambda **kw:pytest.fail('Expired run must not dispatch native'))
    lock=p.run_predictions(output=tmp_path/'out',designs=designs,paths=paths,template={},specs=[],models=models,
        selections=selections,frozen={},policy={'curated_metadata_sources':{}},deadline=time.monotonic()+(-1 if resource=='wall' else 100))
    data=json.loads(lock.read_text());assert data['expected_prediction_rows']==114
    assert data['all_registered_predictions_terminal'] is True and data['completed_native_jobs']==0
    assert data['registered_native_jobs']==8
    assert data['native_execution_attempts']==(3 if resource=='native_failures' else 0)
    assert 'physical_native_jobs' not in data
    config=json.loads((tmp_path/data['config_freeze']['path']).read_text())
    assert len(config['expected_native_jobs'])==8
    statuses=[]; invoked=[]
    for item in data['predictions']:
        predictions=json.loads((tmp_path/item['path']).read_text())
        assert len(predictions['models'])==4 and predictions['selections']==selections
        assert all(r['status'] in {'native_failed','not_run_resource_limit'} and r['concentration'] is None for r in predictions['models'])
        statuses.extend(r['status'] for r in predictions['models'])
        invoked.extend(r['native_execution_invoked'] for r in predictions['models'])
        assert all(type(r['native_execution_invoked']) is bool for r in predictions['models'])
        assert all(r['native_execution_invoked'] is False for r in predictions['models'] if r['status']=='not_run_resource_limit')
    assert statuses.count('native_failed') == (0 if resource=='wall' else 3)
    assert sum(invoked)==data['native_execution_attempts']
    assert len(calls) == (3 if resource=='native_failures' else 0)


def test_scoring_contract_is_literal_and_source_bound(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'ROOT',tmp_path)
    script=tmp_path/'scripts/score_voriconazole_methods_curves.py'
    script.parent.mkdir()
    script.write_text("raise AssertionError('this file must never be imported')\nEVALUATION_CONTRACT={'version':'vori-all-method-curve-evaluation-v1','main':'no outcomes'}\n")
    (tmp_path/'scripts/score_locked_independent_curves.py').write_text('# synthetic helper')
    helper=tmp_path/'src/pbpk_kb/blueprints/design_scoring.py';helper.parent.mkdir(parents=True);helper.write_text('# synthetic metric')
    frozen={};contract=p.load_scoring_contract(frozen)
    assert contract['main']=='no outcomes' and len(frozen)==3
    assert frozen['scripts/score_voriconazole_methods_curves.py']==p.file_sha(script)
