"""Purely fabricated mean curves; no real sealed clinical outcomes are opened."""
from copy import deepcopy
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('aggregate_score_test',ROOT/'scripts/score_locked_aggregate_curves.py')
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return hashlib.sha256(path.read_bytes()).hexdigest()


def records():
    return [{'source_row_ordinal':i+1,'time_h':i,'arithmetic_mean_ng_ml':v,
        'published_arithmetic_mean_is_zero':v==0,
        'zero_interpretation':'source_BLQ_zeroing_or_precision_not_true_zero' if v==0 else None}
        for i,v in enumerate([0,.05,2.])]


def fixture(tmp_path,monkeypatch,*,all_failed=False,native_failed=False,drug='rosuvastatin'):
    monkeypatch.setattr(score,'ROOT',tmp_path)
    program=tmp_path/'score.py';program.write_text('# synthetic program hash');monkeypatch.setattr(score,'__file__',str(program))
    choices=[]
    for i,entry in enumerate(score.expected_selections().values()):
        failed=all_failed or i==0
        choices.append({**entry,'model_key':None if failed else 'model1','prediction_available':not failed,
            'upstream_status':'blocked' if failed else 'budget_exhausted',**({'failure':'synthetic unavailable'} if failed else {})})
    models={} if all_failed else {'model1':{'action':'partition_pksim_standard','values':[1,2],'parameters_path':'params.json'}}
    save(tmp_path/'params.json',{'values':[1,2]})
    case_id='synthetic_'+drug
    censor={'individual_assay_lloq_ng_ml':.1,'apply_individual_lloq_filter_to_positive_group_mean':False}
    sealed={'schema_version':'pbpk-independent-aggregate-curves-sealed-v1','case_id':case_id,'compound':drug,
        'publication_cluster_id':'synthetic_paper','trial_cluster_ids':['synthetic_trial'],'cohort_id':'synthetic_cohort',
        'population_n':32,'individual_records_available':False,'primary_aggregation':score.AGGREGATION,
        'concentration_unit':'ng/mL','time_unit':'h','censoring':censor,'records':records()}
    sealed_path=tmp_path/'sealed/observations.json';sealed_sha=save(sealed_path,sealed)
    design={'schema_version':'pbpk-independent-aggregate-curve-design-v1','case_id':case_id,'compound':drug,
        'publication_cluster_id':'synthetic_paper','trial_cluster_ids':['synthetic_trial'],'cohort_id':'synthetic_cohort',
        'primary_aggregation':score.AGGREGATION,'aggregation_preselected_before_any_model_predictions':True,
        'scientific_results_accessed_by_algorithm':False,'censoring':censor,
        'evidence_exclusion':{'exclude_entire_publication_and_cohort_from_parameter_sources_prompts_retrieval_and_training':True,
            'exclude_entire_trial':True,'raw_source_files':['raw_outcome.xml']},
        'population_metadata':{'n':32,'sex_counts':{'male':14,'female':18}},'baseline_curve_n':32,'prior_paired_endpoint_n':31,
        'protocol':{'sampling_times_min':[0,60,120]},'sampling_times_h':[0,1,2],
        'evaluation_target':{'individual_records_available':False,'scheduled_time_count':3},
        'sealed_observation_reference':{'path':str(sealed_path),'sha256':sealed_sha}}
    design_path=tmp_path/'design.json';design_sha=save(design_path,design)
    qualification={'strict_main_eligible':False,'qualification_tier':'transportability_sensitivity',
        'qualification_reasons':['synthetic unmatched meal, formulation and aggregate estimand']}
    column=f'Organism|PeripheralVenousBlood|{drug.capitalize()}|Plasma (Peripheral Venous Blood) [µmol/l]'
    protocol={'case_id':case_id,'design_sha256':design_sha,'evaluation_contract':score.EVALUATION_CONTRACT,
        'primary_aggregation':score.AGGREGATION,'prediction_aggregation':'sex_weighted_reference_concentration',
        'selections':choices,'models':models,'model_keys':list(models),**qualification,
        'approximation':{'reference_subjects_only':True,'group_mean_reference_mismatch':True,
            'source_individual_BLQ_zeroed_mean_vs_uncensored_prediction':True,'individual_covariates_available':False},
        'sex_weights':{'male':14/32,'female':18/32},'molecular_weight_g_mol':100,
        'sampling_time_min':[0,60,120],'native_concentration_column':column}
    protocol_path=tmp_path/'protocol.json';protocol_sha=save(protocol_path,protocol)
    rows=[]
    for key in models:
        for sex in ('male','female'):
            row={'model_key':key,'sex':sex,'status':'completed','native_execution_invoked':True,
                'time_min':[0,60,120],'concentration':[0,.001,.04],'unit':'µmol/l','time_unit':'min','error':None}
            if native_failed and sex=='female':row.update(status='native_failed',concentration=None,error='synthetic native failure')
            else:
                path=tmp_path/(sex+'.csv')
                with path.open('w',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(['Time [min]',column]);writer.writerows([[0,0],[60,.001],[120,.04]])
                raw=b'<synthetic-pkml/>';archive=tmp_path/(sex+'.pkml.gz');archive.write_bytes(gzip.compress(raw,mtime=0))
                row.update(native_csv_path=str(path),native_csv_sha256=score.sha(path),
                    native_time_column='Time [min]',native_concentration_column=column,
                    pkml_archives=[{'path':str(archive),'sha256':score.sha(archive),'original_path':str(tmp_path/(sex+'.pkml')),
                        'original_sha256':hashlib.sha256(raw).hexdigest(),'original_byte_size':len(raw)}])
            rows.append(row)
    prediction_path=tmp_path/'prediction.json';prediction_sha=save(prediction_path,{'case_id':case_id,
        'models':rows,'selections':choices,'protocol_sha256':protocol_sha})
    case={'case_id':case_id,'design_path':str(design_path),'design_sha256':design_sha,
        'protocol_path':str(protocol_path),'protocol_sha256':protocol_sha}
    config={'cases':[case],'selections':choices,'models':models,'evaluation_contract':score.EVALUATION_CONTRACT,
        **qualification,'frozen_input_sha256':{str(program):score.sha(program),str(design_path):design_sha},
        'expected_native_jobs':[{'case_id':case_id,'model_key':k,'sex':s} for k in models for s in ('male','female')]}
    config_path=tmp_path/'freeze.json';config_sha=save(config_path,config)
    lock={'schema_version':'pbpk-external-predictions-lock-v1','model_selection_frozen':True,
        'all_registered_predictions_terminal':True,'method_job_count':57,'expected_prediction_rows':57,
        'config_freeze':{'path':str(config_path),'sha256':config_sha},
        'predictions':[{**case,'path':str(prediction_path),'sha256':prediction_sha}],
        'registered_native_jobs':len(rows),'native_execution_attempts':len(rows),
        'completed_native_jobs':sum(r['status']=='completed' for r in rows)}
    path=tmp_path/'lock.json';save(path,lock)
    return path,lock,config


def rewrite_prediction(path,lock,mutate):
    item=lock['predictions'][0];prediction=json.loads(Path(item['path']).read_text())
    mutate(prediction);item['sha256']=save(Path(item['path']),prediction);save(path,lock)


def rewrite_config(path,lock,config):
    lock['config_freeze']['sha256']=save(Path(lock['config_freeze']['path']),config);save(path,lock)


def forbid_outcomes(monkeypatch):
    real_read,real_sha=score.read,score.sha
    def guard(path):
        if 'sealed' in Path(path).parts or Path(path).name=='raw_outcome.xml':pytest.fail('Outcome bytes accessed before global lock validation')
    def read(path):guard(path);return real_read(path)
    def sha(path):guard(path);return real_sha(path)
    monkeypatch.setattr(score,'read',read);monkeypatch.setattr(score,'sha',sha)


def test_positive_published_mean_below_individual_lloq_is_included():
    result=score.curve_metrics(records(),[0,60,120],[0,.1,4.])
    assert result['positive_mean_points']==2 and result['points'][1]['included_in_log_error']
    assert result['log_rmse']==pytest.approx(math.log(2))
    assert result['gmfe']==pytest.approx(2) and result['fold_bias']==pytest.approx(2)
    assert result['within_2fold_fraction']==1 and result['individual_lloq_filter_applied'] is False
    assert result['points'][0]['log_exclusion_reason']=='source_zeroed_or_rounded_mean_log_undefined'
    endpoints=result['mean_curve_endpoints'];assert endpoints['domain_complete'] and endpoints['not_mean_individual_NCA']
    assert endpoints['sampled_peak_of_reported_mean']['reported_mean_target']==2
    assert endpoints['observed_window_auc_of_reported_mean']['reported_mean_target']==pytest.approx(63)
    assert endpoints['observed_window_auc_of_reported_mean']['reference_prediction']==pytest.approx(126)


def test_missing_rows_are_retained_and_do_not_allow_endpoint_window_selection():
    rows=records();rows[1].update(arithmetic_mean_ng_ml=None,arithmetic_mean_missing_reason='source_missing_unspecified')
    result=score.curve_metrics(rows,[0,60,120],[0,.1,4])
    assert len(result['points'])==3 and result['published_values_available']==2
    assert result['mean_curve_endpoints']['sampled_peak_of_reported_mean'] is None
    assert result['mean_curve_endpoints']['observed_window_auc_of_reported_mean'] is None


def test_nonpositive_predictions_at_positive_mean_are_failure_not_clipped():
    result=score.curve_metrics(records(),[0,60,120],[0,0,4])
    assert result['status']=='failed_retained_in_denominator' and result['log_rmse'] is None
    with pytest.raises(ValueError):score.curve_metrics(records(),[0,60,120],[0,-.1,4])


@pytest.mark.parametrize('drug',['atorvastatin','rosuvastatin'])
def test_complete_validation_of_both_statin_interfaces_never_reads_outcomes(tmp_path,monkeypatch,drug):
    path,_,_=fixture(tmp_path,monkeypatch,drug=drug);forbid_outcomes(monkeypatch)
    _,config,cases=score.validate_locked_inputs(path)
    assert len(config['selections'])==57 and len(cases)==1


@pytest.mark.parametrize('damage',['prediction_hash','missing_sex','negative_sex','negative_native_tail','sampling','csv_binding','selection','archive'])
def test_any_lock_damage_prevents_all_sealed_reads(tmp_path,monkeypatch,damage):
    path,lock,_=fixture(tmp_path,monkeypatch);forbid_outcomes(monkeypatch)
    if damage=='prediction_hash':Path(lock['predictions'][0]['path']).write_text('{}')
    else:
        def change(pred):
            if damage=='missing_sex':pred['models'].pop()
            elif damage=='negative_sex':pred['models'][0]['concentration'][1]=-.001
            elif damage=='negative_native_tail':
                row=pred['models'][0];csv_path=Path(row['native_csv_path'])
                with csv_path.open('a') as stream:stream.write('180,-0.00001\n')
                row['native_csv_sha256']=score.sha(csv_path)
            elif damage=='sampling':pred['models'][0]['time_min'][1]=61
            elif damage=='csv_binding':pred['models'][0]['concentration'][1]=.1
            elif damage=='selection':pred['selections'][1]['solver_budget']=2000
            else:pred['models'][0]['pkml_archives'][0]['original_sha256']='0'*64
        rewrite_prediction(path,lock,change)
    with pytest.raises(ValueError):score.score_locked(path,tmp_path/'scores')


@pytest.mark.parametrize('damage',['missing_method','wrong_tier','wrong_counts','metric_change','sealed_source','raw_source'])
def test_prefrozen_scope_qualification_and_denied_inputs_cannot_change(tmp_path,monkeypatch,damage):
    path,lock,config=fixture(tmp_path,monkeypatch);forbid_outcomes(monkeypatch)
    if damage=='missing_method':config['selections'].pop()
    elif damage=='wrong_tier':config['strict_main_eligible']=True
    elif damage=='wrong_counts':lock['native_execution_attempts']=0
    elif damage=='metric_change':config['evaluation_contract']={'version':'posthoc'}
    elif damage=='sealed_source':config['frozen_input_sha256'][str(tmp_path/'sealed/observations.json')]='0'*64
    else:config['frozen_input_sha256']['raw_outcome.xml']='0'*64
    rewrite_config(path,lock,config)
    with pytest.raises(ValueError):score.score_locked(path,tmp_path/'scores')


def test_failed_methods_preserve_57_rows_and_scientific_tier_without_fake_individuals(tmp_path,monkeypatch):
    path,_,_=fixture(tmp_path,monkeypatch);receipt=score.score_locked(path,tmp_path/'scores')
    assert receipt['method_scenario_rows_evaluated']==56 and receipt['method_scenario_rows_failed_retained']==1
    assert receipt['source_cohort_n']==32 and receipt['aggregate_curves']==1
    assert receipt['independent_drugs']==receipt['independent_studies']==1 and receipt['independent_individual_curves']==0
    assert receipt['strict_main_eligible'] is False and receipt['qualification_tier']=='transportability_sensitivity'
    result=json.loads((tmp_path/receipt['sealed_scores_path']).read_text())
    assert len(result['method_scenarios'])==57 and len(result['method_budget_strata'])==21
    assert all(r['confidence_interval'] is None for r in result['method_budget_strata'])
    assert 'log_rmse' not in json.dumps(receipt)


@pytest.mark.parametrize('all_failed,native_failed',[(True,False),(False,True)])
def test_all_upstream_or_native_failures_remain_in_denominator(tmp_path,monkeypatch,all_failed,native_failed):
    path,_,_=fixture(tmp_path,monkeypatch,all_failed=all_failed,native_failed=native_failed)
    receipt=score.score_locked(path,tmp_path/'scores')
    assert receipt['method_scenario_rows_failed_retained']==57 and receipt['method_scenario_rows_evaluated']==0


def test_full_baseline_sex_weights_cannot_be_replaced_with_paired_endpoint_n(tmp_path,monkeypatch):
    path,lock,config=fixture(tmp_path,monkeypatch);forbid_outcomes(monkeypatch)
    proto_path=Path(lock['predictions'][0]['protocol_path']);proto=json.loads(proto_path.read_text())
    proto['sex_weights']={'male':14/31,'female':17/31};newsha=save(proto_path,proto)
    lock['predictions'][0]['protocol_sha256']=newsha;config['cases'][0]['protocol_sha256']=newsha
    rewrite_prediction(path,lock,lambda pred:pred.update(protocol_sha256=newsha));rewrite_config(path,lock,config)
    with pytest.raises(ValueError,match='full baseline cohort'):score.score_locked(path,tmp_path/'scores')


def test_production_rosuvastatin_writer_lock_is_consumed_without_sealed_reads(tmp_path,monkeypatch):
    """Production writer/archiver, fabricated native process, synthetic design."""
    _,_,configuration=fixture(tmp_path,monkeypatch)
    predictor_spec=importlib.util.spec_from_file_location('rosu_aggregate_writer_test',ROOT/'scripts/predict_rosuvastatin_methods_external.py')
    predictor=importlib.util.module_from_spec(predictor_spec);predictor_spec.loader.exec_module(predictor)
    monkeypatch.setattr(predictor,'ROOT',tmp_path);monkeypatch.setattr(predictor,'STAGE',tmp_path/'stage')
    script=tmp_path/'scripts/score_locked_aggregate_curves.py';script.parent.mkdir()
    script.write_bytes((ROOT/'scripts/score_locked_aggregate_curves.py').read_bytes())
    design_path=tmp_path/'design.json';design=json.loads(design_path.read_text())
    design['population_metadata']['sex_weights']={'male':14/32,'female':18/32};save(design_path,design)
    monkeypatch.setattr(predictor,'make_request',lambda *a:{'synthetic_request':True})
    monkeypatch.setattr(predictor,'audit_design_inputs',lambda *a,**kw:None)
    monkeypatch.setattr(predictor,'compile_native_candidate',lambda *a,**kw:{'snapshot':{'synthetic':True}})
    def native(**kwargs):
        directory=kwargs['output_dir'];directory.mkdir(parents=True)
        curve=directory/'curve.csv'
        with curve.open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['Time [min]',predictor.CONCENTRATION_COLUMN]);writer.writerows([[0,0],[60,.001],[120,.04]])
        pkml=directory/'model.pkml';pkml.write_bytes(b'<purely-synthetic-model/>')
        result={'engineering_accepted':True,'curve_outputs':[{'path':'curve.csv','sha256':predictor.file_sha(curve)}],
            'pkml_outputs':[{'path':'model.pkml','sha256':predictor.file_sha(pkml),'byte_size':pkml.stat().st_size}]}
        save(directory/'execution_manifest.json',result);return result
    monkeypatch.setattr(predictor,'execute_snapshot',native)
    template={'instance':{'compound':{'Parameters':[{'Name':'Molecular weight','Value':100}]},
        'individual':{'OriginData':{'Population':'synthetic'}},'protocol':{'synthetic':'clinical source proxy'},
        'formulation':{'synthetic':'source release'}}}
    lock=predictor.run_predictions(output=tmp_path/'production_writer',designs={40:design},paths={40:design_path},
        template=template,specs=[],models=configuration['models'],selections=configuration['selections'],
        frozen={str(design_path.relative_to(tmp_path)):predictor.file_sha(design_path)},
        policy={'curated_metadata_sources':{}},deadline=time.monotonic()+60)
    forbid_outcomes(monkeypatch)
    _,frozen,cases=score.validate_locked_inputs(lock)
    assert len(frozen['selections'])==57 and len(cases)==1
    assert cases[0]['qualification']['strict_main_eligible'] is False
    assert not cases[0]['models']['model1']['native_failures']
