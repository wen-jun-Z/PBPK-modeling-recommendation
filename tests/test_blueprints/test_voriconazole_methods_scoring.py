"""Only fabricated curves and subjects; never load project clinical outcomes."""
import csv
import gzip
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('vori_methods_score_test',ROOT/'scripts/score_voriconazole_methods_curves.py')
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path,monkeypatch,*,all_failed=False,one_failed=True,native_failed=False):
    monkeypatch.setattr(score,'ROOT',tmp_path)
    program=tmp_path/'scorer.py';program.write_text('# synthetic scorer identity')
    monkeypatch.setattr(score,'__file__',str(program))
    selections=[]
    for i,entry in enumerate(score.expected_selections().values()):
        failed=all_failed or (i==0 and one_failed)
        selections.append({**entry,'model_key':None if failed else 'modelA','prediction_available':not failed,
            'upstream_status':'blocked' if failed else 'budget_exhausted',
            **({'failure':'synthetic upstream failure'} if failed else {})})
    models={} if all_failed else {'modelA':{'action':'partition_pksim_standard','values':[1,2],
        'parameters_path':'parameters.json','parameter_contract_sha256':'e'*64}}
    cases=[];records=[];sources={str(program):score.sha(program)}
    for index in (4,6):
        case_id=f'synthetic_{index}';directory=tmp_path/str(index)
        sealed_path=tmp_path/'sealed'/f'{case_id}.json'
        subjects=[{'subject_id':f'{index}_{person}','sequence':'TR','period':2,'treatment':'R',
            'observations':[
                {'nominal_time_h':0,'concentration_ng_ml':None,'observation_status':'left_censored_below_LLOQ','source_token':'BLQ'},
                {'nominal_time_h':1.33,'concentration_ng_ml':100,'observation_status':'quantified','source_token':'100'}]}
            for person in (1,2)]
        sealed_sha=save(sealed_path,{'case_id':case_id,'publication_cluster_id':'synthetic_study','unit':'ng/ml',
            'assay_LLOQ_ng_ml':10,'records':subjects})
        design={'case_id':case_id,'schema_version':'pbpk-independent-curve-design-v1','compound':'synthetic_drug',
            'publication_cluster_id':'synthetic_study','cohort_id':f'dose{index}',
            'population':{'n':2,'male_n':1,'female_n':1},'sampling':{'time_h':[0,1.33]},
            'sealed_observation_reference':{'path':str(sealed_path),'sha256':sealed_sha},
            'curve_evaluation_contract':{'assay_LLOQ_ng_ml':10},
            'crossover':{'all_subject_links':[{'subject_id':r['subject_id'],'sequence':'TR','periods':{'R':2}} for r in subjects]}}
        design_path=directory/'design.json';design_sha=save(design_path,design);sources[str(design_path)]=design_sha
        protocol={'case_id':case_id,'design_sha256':design_sha,'prediction_aggregation':'sex_weighted_reference_concentration',
            'evaluation_contract':score.EVALUATION_CONTRACT,
            'selections':selections,'models':models,'model_keys':list(models),'sex_weights':{'male':.5,'female':.5},
            'molecular_weight_g_mol':100,'sampling_time_min':[0,79.8]}
        protocol_path=directory/'protocol.json';protocol_sha=save(protocol_path,protocol)
        prediction={'case_id':case_id,'protocol_sha256':protocol_sha,'selections':selections,'models':[]}
        for model in models:
            for sex in ('male','female'):
                row={'model_key':model,'sex':sex,'status':'completed','native_execution_invoked':True,'time_min':[0,79.8],
                    'concentration':[0,1],'unit':'µmol/l','time_unit':'min','error':None}
                if native_failed and index==6 and sex=='female':
                    row.update(status='native_failed',concentration=None,error='synthetic native failure')
                else:
                    native=directory/(sex+'.csv')
                    with native.open('w',newline='') as stream:
                        writer=csv.writer(stream);writer.writerow(['Time [min]',score.TARGET]);writer.writerows([[0,0],[79.8,1]])
                    row.update(native_csv_path=str(native),native_csv_sha256=score.sha(native))
                    pkml=b'<synthetic-native-model/>';archive=directory/(sex+'.pkml.gz')
                    archive.write_bytes(gzip.compress(pkml,mtime=0))
                    row['pkml_archives']=[{'path':str(archive),'sha256':score.sha(archive),
                        'original_path':str(directory/(sex+'.pkml')),'original_sha256':hashlib.sha256(pkml).hexdigest(),
                        'original_byte_size':len(pkml)}]
                prediction['models'].append(row)
        pred_path=directory/'prediction.json';pred_sha=save(pred_path,prediction)
        case={'case_id':case_id,'design_path':str(design_path),'design_sha256':design_sha,
            'protocol_path':str(protocol_path),'protocol_sha256':protocol_sha}
        cases.append(case);records.append({**case,'path':str(pred_path),'sha256':pred_sha})
    freeze={'cases':cases,'selections':selections,'models':models,'frozen_input_sha256':sources,'evaluation_contract':score.EVALUATION_CONTRACT,
        'expected_native_jobs':[{'case_id':c['case_id'],'model_key':m,'sex':s} for c in cases for m in models for s in ('male','female')]}
    freeze_path=tmp_path/'freeze.json';freeze_sha=save(freeze_path,freeze)
    lock={'schema_version':'pbpk-external-predictions-lock-v1','model_selection_frozen':True,
        'all_registered_predictions_terminal':True,'method_job_count':57,'expected_prediction_rows':114,
        'registered_native_jobs':len(models)*4,'native_execution_attempts':len(models)*4,
        'completed_native_jobs':len(models)*4-int(native_failed and bool(models)),
        'config_freeze':{'path':str(freeze_path),'sha256':freeze_sha},'predictions':records}
    lock_path=tmp_path/'lock.json';save(lock_path,lock)
    return lock_path,lock,freeze


def rewrite_prediction(lock_path,lock,index,mutate):
    item=lock['predictions'][index];path=Path(item['path']);prediction=json.loads(path.read_text())
    mutate(prediction);item['sha256']=save(path,prediction);save(lock_path,lock)


def rewrite_freeze(lock_path,lock,freeze):
    lock['config_freeze']['sha256']=save(Path(lock['config_freeze']['path']),freeze);save(lock_path,lock)


def forbid_sealed(monkeypatch):
    original=score.read
    def safe_read(path):
        if 'sealed' in Path(path).parts:pytest.fail('A lock failure must precede every sealed read')
        return original(path)
    monkeypatch.setattr(score,'read',safe_read)


def test_complete_validation_reads_no_outcomes_and_keeps_nominal_1_33h(tmp_path,monkeypatch):
    path,_,_=fixture(tmp_path,monkeypatch);forbid_sealed(monkeypatch)
    _,config,cases=score.validate_locked_inputs(path)
    assert len(config['selections'])==57 and len(cases)==2
    assert cases[0]['time_min'][1]==pytest.approx(79.8)


@pytest.mark.parametrize('damage',['hash','grid','missing_sex','native_csv','changed_selection'])
def test_second_case_damage_stops_before_even_first_case_outcomes(tmp_path,monkeypatch,damage):
    path,lock,_=fixture(tmp_path,monkeypatch);forbid_sealed(monkeypatch)
    if damage=='hash':Path(lock['predictions'][1]['path']).write_text('{}')
    elif damage=='grid':rewrite_prediction(path,lock,1,lambda p:p['models'][0]['time_min'].__setitem__(1,80))
    elif damage=='missing_sex':rewrite_prediction(path,lock,1,lambda p:p['models'].pop())
    elif damage=='native_csv':rewrite_prediction(path,lock,1,lambda p:p['models'][0]['concentration'].__setitem__(1,2))
    else:rewrite_prediction(path,lock,1,lambda p:p['selections'][0].update(solver_budget=2000))
    with pytest.raises(ValueError):score.score_locked(path,tmp_path/'scores')


def test_no_sealed_file_may_be_sneaked_into_frozen_sources(tmp_path,monkeypatch):
    path,lock,freeze=fixture(tmp_path,monkeypatch);forbid_sealed(monkeypatch)
    design=json.loads(Path(freeze['cases'][0]['design_path']).read_text());ref=design['sealed_observation_reference']
    freeze['frozen_input_sha256'][ref['path']]=ref['sha256'];rewrite_freeze(path,lock,freeze)
    with pytest.raises(ValueError,match='Sealed observations'):score.score_locked(path,tmp_path/'scores')


def test_all_57_selections_and_failed_denominators_are_retained(tmp_path,monkeypatch):
    path,_,_=fixture(tmp_path,monkeypatch)
    result=score.score_locked(path,tmp_path/'scores')
    assert result['method_dose_rows_evaluated']==112 and result['method_dose_rows_failed_retained']==2
    assert result['independent_drugs']==result['independent_studies']==1
    scores=json.loads((tmp_path/result['sealed_scores_path']).read_text())
    assert len(scores['method_cases'])==114 and len(scores['method_studies'])==57
    assert len(scores['method_budget_strata'])==21 and len(scores['unique_model_cases'])==2
    assert all(r['study_log_rmse']==pytest.approx(0) for r in scores['method_studies'] if r['status']=='evaluated')
    assert 'study_log_rmse' not in json.dumps(result)


def test_native_failure_is_not_replaced_by_successful_sex_or_other_dose(tmp_path,monkeypatch):
    path,_,_=fixture(tmp_path,monkeypatch,native_failed=True)
    result=score.score_locked(path,tmp_path/'scores')
    scores=json.loads((tmp_path/result['sealed_scores_path']).read_text())
    assert result['method_dose_rows_failed_retained']==58
    assert all(r['study_log_rmse'] is None for r in scores['method_studies'])
    assert all(r['mean_across_registered_seeds'] is None for r in scores['method_budget_strata'])


def test_all_failed_upstream_still_produces_full_denominator(tmp_path,monkeypatch):
    path,_,_=fixture(tmp_path,monkeypatch,all_failed=True)
    result=score.score_locked(path,tmp_path/'scores')
    assert result['method_dose_rows_failed_retained']==114 and result['unique_models']==0


@pytest.mark.parametrize('change',['drop_method','budget','duplicate_job'])
def test_frozen_scope_must_be_complete_and_stratified(tmp_path,monkeypatch,change):
    path,lock,freeze=fixture(tmp_path,monkeypatch);forbid_sealed(monkeypatch)
    if change=='drop_method':freeze['selections'].pop()
    elif change=='budget':freeze['selections'][1]['solver_budget']=2000
    else:freeze['expected_native_jobs'][1]=deepcopy(freeze['expected_native_jobs'][0])
    rewrite_freeze(path,lock,freeze)
    with pytest.raises(ValueError):score.score_locked(path,tmp_path/'scores')


def test_nonpositive_prediction_fails_without_clipping_or_subject_exclusion(tmp_path,monkeypatch):
    path,lock,_=fixture(tmp_path,monkeypatch,one_failed=False)
    for index in range(2):
        item=lock['predictions'][index];prediction=json.loads(Path(item['path']).read_text())
        for row in prediction['models']:
            row['concentration'][1]=0
            native=Path(row['native_csv_path'])
            with native.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['Time [min]',score.TARGET]);writer.writerows([[0,0],[79.8,0]])
            row['native_csv_sha256']=score.sha(native)
        item['sha256']=save(Path(item['path']),prediction)
    save(path,lock)
    result=score.score_locked(path,tmp_path/'scores')
    assert result['method_dose_rows_failed_retained']==114


@pytest.mark.parametrize('only_native_tail',[False,True])
def test_negative_sex_concentration_cannot_be_hidden_by_other_sex(tmp_path,monkeypatch,only_native_tail):
    path,lock,_=fixture(tmp_path,monkeypatch,one_failed=False);forbid_sealed(monkeypatch)
    def alter(prediction):
        row=prediction['models'][0]
        if only_native_tail:
            native=Path(row['native_csv_path'])
            with native.open('a') as stream:stream.write('120,-0.001\n')
            row['native_csv_sha256']=score.sha(native)
        else:row['concentration'][1]=-0.5
    rewrite_prediction(path,lock,1,alter)
    with pytest.raises(ValueError):score.score_locked(path,tmp_path/'scores')


def test_rehashed_archive_must_preserve_exact_original_pkml_bytes(tmp_path,monkeypatch):
    path,lock,_=fixture(tmp_path,monkeypatch);forbid_sealed(monkeypatch)
    def alter(prediction):
        record=prediction['models'][0]['pkml_archives'][0];archive=Path(record['path'])
        archive.write_bytes(gzip.compress(b'not-the-same-native-model',mtime=0));record['sha256']=score.sha(archive)
    rewrite_prediction(path,lock,1,alter)
    with pytest.raises(ValueError,match='Archive|archive'):score.score_locked(path,tmp_path/'scores')


def test_auxiliary_error_and_strict_complete_grid_endpoints_have_fixed_definitions():
    record={'subject_id':'fabricated','sequence':'TR','period':2,'treatment':'R',
        'observations':[{'nominal_time_h':i,'concentration_ng_ml':v,'observation_status':'quantified','source_token':str(v)}
            for i,v in enumerate([100,200,100])]}
    result=score.score_subject_metrics(record,[0,60,120],[2,4,2],100,10)
    assert result['gmfe']==pytest.approx(2) and result['fold_bias']==pytest.approx(2)
    assert result['within_2fold_fraction']==1
    endpoints=result['strict_observed_window_endpoints'];assert endpoints['evaluable']
    assert endpoints['sampled_cmax']['fold_ratio']==pytest.approx(2)
    assert endpoints['observed_window_auc']['observed']==pytest.approx(180)
    assert endpoints['observed_window_auc']['predicted']==pytest.approx(360)
    record['observations'][1].update(concentration_ng_ml=None,observation_status='source_missing_unspecified',source_token='-')
    result=score.score_subject_metrics(record,[0,60,120],[2,4,2],100,10)
    assert result['status']=='evaluated' and result['gmfe']==pytest.approx(2)
    assert result['strict_observed_window_endpoints']['evaluable'] is False
    assert result['strict_observed_window_endpoints']['observed_window_auc'] is None
    assert result['strict_observed_window_endpoints']['missing_domain']==[{'source_index':1,'reason':'source_missing_token_not_imputed'}]


def test_metric_contract_cannot_change_after_prediction(tmp_path,monkeypatch):
    path,lock,freeze=fixture(tmp_path,monkeypatch);forbid_sealed(monkeypatch)
    freeze['evaluation_contract']={'version':'posthoc_metric_change'};rewrite_freeze(path,lock,freeze)
    with pytest.raises(ValueError,match='metric definitions'):score.score_locked(path,tmp_path/'scores')


def test_censored_peak_is_exact_and_auc_bounds_contain_every_permitted_assignment():
    record={'observations':[
        {'observation_status':'left_censored_below_LLOQ','concentration_ng_ml':None},
        {'observation_status':'quantified','concentration_ng_ml':200},
        {'observation_status':'left_censored_below_LLOQ','concentration_ng_ml':None}]}
    result=score.censored_pk_endpoints(record,[0,60,120],[0,2,0],100,10)
    assert result['sampled_cmax']['observed']==2
    assert result['sampled_cmax']['exact_sampled_peak_despite_other_left_censored_positions']
    auc=result['observed_window_auc_interval']
    assert auc['observed_lower']==120 and auc['observed_upper']==126
    for first in [0,.025,.1]:
        for last in [0,.05,.1]:
            actual=30*(first+2)+30*(2+last)
            assert auc['observed_lower']<=actual<=auc['observed_upper']
    assert auc['within_observed_interval'] and auc['within_interval_is_not_precise_accuracy']
    record['observations'][0]['observation_status']='source_missing_unspecified'
    result=score.censored_pk_endpoints(record,[0,60,120],[0,2,0],100,10)
    assert result['sampled_cmax'] is None and result['observed_window_auc_interval'] is None
    assert result['missing_positions']==[0]


def test_censored_auxiliary_refuses_negative_predictions_and_retains_interval_width():
    record={'observations':[{'observation_status':'left_censored_below_LLOQ','concentration_ng_ml':None}]*2}
    result=score.censored_pk_endpoints(record,[0,60],[-.1,1],100,10)
    assert result['reason']=='invalid_native_prediction_or_grid'
    result=score.censored_pk_endpoints(record,[0,60],[.05,.05],100,10)
    assert result['sampled_cmax'] is None
    assert result['observed_window_auc_interval']['observed_interval_width']==6
    assert result['observed_window_auc_interval']['within_observed_interval']


def test_real_prediction_writer_contract_passes_scorer_without_any_outcome_read(tmp_path,monkeypatch):
    """Use production lock/archive/CSV writing with a fabricated native process."""
    _,_,fixture_config=fixture(tmp_path,monkeypatch)
    predictor_spec=importlib.util.spec_from_file_location('vori_predictor_wire_test',ROOT/'scripts/predict_voriconazole_methods_external.py')
    predictor=importlib.util.module_from_spec(predictor_spec);predictor_spec.loader.exec_module(predictor)
    monkeypatch.setattr(predictor,'ROOT',tmp_path)
    monkeypatch.setattr(predictor,'STAGE',tmp_path/'stage')
    scripts=tmp_path/'scripts';scripts.mkdir()
    (scripts/'score_voriconazole_methods_curves.py').write_bytes((ROOT/'scripts/score_voriconazole_methods_curves.py').read_bytes())
    (scripts/'score_locked_independent_curves.py').write_text('# synthetic hash-only helper')
    helper=tmp_path/'src/pbpk_kb/blueprints/design_scoring.py';helper.parent.mkdir(parents=True);helper.write_text('# synthetic metric identity')
    save(tmp_path/'parameters.json',{'values':[1,2]})
    designs={};paths={};frozen={}
    for dose,case in zip((4,6),fixture_config['cases']):
        path=Path(case['design_path']);design=json.loads(path.read_text())
        design['population'].update(n=24,male_n=12,female_n=12)
        design['dose']={'value':dose,'infusion_duration_min':dose*20}
        save(path,design);designs[dose]=design;paths[dose]=path;frozen[str(path.relative_to(tmp_path))]=predictor.file_sha(path)
    monkeypatch.setattr(predictor,'make_request',lambda *a:{'synthetic_request':True})
    monkeypatch.setattr(predictor,'audit_design_inputs',lambda *a,**kw:None)
    monkeypatch.setattr(predictor,'compile_native_candidate',lambda *a,**kw:{'snapshot':{'synthetic':True}})
    def execute(**kwargs):
        directory=kwargs['output_dir'];directory.mkdir(parents=True)
        curve=directory/'curve.csv'
        with curve.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['Time [min]',score.TARGET]);writer.writerows([[0,0],[79.8,1],[120,.5]])
        pkml=directory/'model.pkml';pkml.write_bytes(b'<purely-synthetic-pkml/>')
        result={'engineering_accepted':True,'curve_outputs':[{'path':'curve.csv','sha256':predictor.file_sha(curve)}],
            'pkml_outputs':[{'path':'model.pkml','sha256':predictor.file_sha(pkml),'byte_size':pkml.stat().st_size}]}
        save(directory/'execution_manifest.json',result);return result
    monkeypatch.setattr(predictor,'execute_snapshot',execute)
    template={'instance':{'compound':{'Parameters':[{'Name':'Molecular weight','Value':100}]},
        'individual':{'OriginData':{'Population':'synthetic'}}}}
    lock=predictor.run_predictions(output=tmp_path/'production_writer_output',designs=designs,paths=paths,
        template=template,specs=[],models=fixture_config['models'],selections=fixture_config['selections'],
        frozen=frozen,policy={'curated_metadata_sources':{}},deadline=time.monotonic()+60)
    forbid_sealed(monkeypatch)
    _,configuration,checked=score.validate_locked_inputs(lock)
    assert len(configuration['selections'])==57 and len(checked)==2
    assert all(c['models']['modelA']['status']=='available' for c in checked)
