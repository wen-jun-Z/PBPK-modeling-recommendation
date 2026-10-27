"""Separated-study CLI contracts and prediction locking; no native computation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('method_cli',ROOT/'scripts/run_blueprint_experiments.py')
cli=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    external=tmp_path/'external.json'
    external.write_text(json.dumps({'scenario_id':'test','compound_id':'drug','study_group_id':'study_test',
        'role':'test','observations':[{'time_min':1,'concentration_umol_per_l':1}]}))
    scenario={'scenario_id':'test','study_group_id':'study_test',
              'holdout_reference':{'path':'external.json','sha256':cli.file_sha(external)}}
    case={'case_id':'case','compound_id':'drug','tier':'development',
          'scenarios':[{'scenario_id':'cal','study_group_id':'study_cal'},
                       {'scenario_id':'sel','study_group_id':'study_sel'}]}
    payload={'case':case,'external_scenarios':[scenario]}
    path=tmp_path/'input_package.json';path.write_text(json.dumps(payload))
    return path,payload


@pytest.mark.parametrize('contamination',['study','scenario','values'])
def test_external_leakage_rejected_before_native_execution(bundle,contamination):
    path,payload=bundle
    external=payload['external_scenarios'][0]
    if contamination=='study': external['study_group_id']='study_cal'
    elif contamination=='scenario': external['scenario_id']='sel'
    else: external['observations']=[{'concentration_umol_per_l':1.}]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):cli.load_case_bundle(path)


def test_every_methods_prediction_is_locked_before_external_values_are_read(bundle,monkeypatch):
    path,payload=bundle;out=path.parent/'output';out.mkdir()
    predicted=[]
    def predict(candidate,scenario,points,context,phase):
        predicted.append(candidate['candidate_id']);context['ledger'].consume()
        return [{'vector_index':0,'time_min':[0.,1.,2.],'concentration':[0.,1.,2.],
                 'time_unit':'min','unit':'umol/l'}]
    experiment=SimpleNamespace(_predict=predict,device='cpu')
    chosen=[{'method':m,'seed':17,'candidate':{'candidate_id':m},'result':{'fitted_values':[1.]},
             'directory':out/'methods'/f'{m}_seed_17'} for m in ('M0','M1')]
    original=Path.read_text
    def read_text(path,*args,**kwargs):
        if path.name=='external.json':
            assert predicted==['M0','M1']
            assert (out/'external_prediction_locks.json').is_file()
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read_text)
    count=cli.evaluate_external(experiment,payload['case'],payload['external_scenarios'],chosen,out,lambda **kwargs:None)
    assert count==2
    assert (out/'external_results.json').is_file()


@pytest.mark.parametrize('successful_status',['completed','budget_exhausted'])
def test_deferred_prediction_keeps_blocked_method_and_prelocked_base(bundle,monkeypatch,successful_status):
    path,payload=bundle;out=path.parent/'deferred';out.mkdir()
    payload['case']['parameter_search']={'parameters':[{'initial_value':1.}]}
    curve={'vector_index':0,'time_min':[0.,1.,2.],'concentration':[0.,1.,2.],
           'time_unit':'min','unit':'umol/l'}
    calls=[]
    def predict(candidate,scenario,points,context,phase):
        calls.append(candidate['candidate_id']);context['ledger'].consume();return [curve]
    experiment=SimpleNamespace(_predict=predict,device='cpu')
    chosen=[{'method':'M0','seed':17,'candidate':{'candidate_id':'selected'},
             'result':{'fitted_values':[2.]},'directory':out/'methods/M0_seed_17'}]
    original=Path.read_text
    def forbid_observation_read(target,*args,**kwargs):
        if target.name=='external.json':raise AssertionError('Outcomes decoded during deferred predictions')
        return original(target,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',forbid_observation_read)
    records=cli.predict_external(experiment,payload['case'],payload['external_scenarios'],chosen,out,
        lambda **kwargs:None,jobs=[{'method':'M0','seed':17,'status':successful_status},
                                  {'method':'M2','seed':17,'status':'blocked','blocked_reason':'unavailable'}],
        fallback_candidate={'candidate_id':'base'})
    assert calls==['base','selected']
    assert records[0]['locked_selection']['status']==successful_status
    assert records[1]['curves']=={'test':None}
    assert records[1]['locked_selection']['status']=='blocked'
    monkeypatch.setattr(Path,'read_text',original)
    assert cli.score_external(payload['case'],payload['external_scenarios'],records,out,lambda **kwargs:None,device='cpu')==2
    report=json.loads((out/'methods/M2_seed_17/external_evaluation.json').read_text())
    assert report['rows'][0]['status']=='scored_with_fallback'
    assert report['rows'][0]['fallback_used'] is True
    assert report['rows'][0]['log_rmse']==0.


def test_prespecified_endpoint_eligibility_reaches_scoring_without_changing_measurements(bundle,monkeypatch):
    path,payload=bundle;out=path.parent/'endpoints';out.mkdir()
    scene=payload['external_scenarios'][0]
    scene.update(auc_evaluable=True,cmax_evaluable=True,sampling_qualification={'scope':'observed_window'})
    before=(path.parent/'external.json').read_bytes()
    captured=[]
    def score(observations,*args,**kwargs):
        captured.extend(observations);return {'rows':[]}
    from pbpk_kb.blueprints import heldout_evaluation
    monkeypatch.setattr(heldout_evaluation,'evaluate_locked_predictions',score)
    record={'method':'M0','seed':17,'curves':{},'locked_selection':{}}
    cli.score_external(payload['case'],[scene],[record],out,lambda **kw:None,device='cpu')
    assert captured[0]['auc_evaluable'] is True and captured[0]['cmax_evaluable'] is True
    assert captured[0]['observations']==json.loads(before)['observations']
    assert (path.parent/'external.json').read_bytes()==before
