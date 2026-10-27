#!/usr/bin/env python3
"""Seal all-method errors against a published arithmetic group-mean PK curve.

This compares a registered reference-population approximation with the source
BLQ-zeroed mean. It never fabricates individual records or individual NCA.
"""
from __future__ import annotations
import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
AGGREGATION='arithmetic_mean_of_source_BLQ_zeroed_individual_concentrations'
BLOCKS={
    'main_5000':(5000,None,('M0','M1','M2','M3','M4','M5','M6','A1','A2','A3','E0')),
    'budget_500':(500,None,('M3','M4','M5','M6')),
    'budget_2000':(2000,None,('M3','M4','M5','M6')),
    'matched_1':(5000,1,('M5','M6')),
}
EVALUATION_CONTRACT={
    'version':'pbpk-aggregate-curve-evaluation-v1',
    'target':'published arithmetic mean after the source set individual BLQ concentrations to zero',
    'primary_domain':'all strictly positive published arithmetic mean positions, including values below individual assay LLOQ',
    'zero_policy':'retain published zero positions and source censoring/precision reason; exclude only from logarithmic errors, never replace with epsilon',
    'log_rmse':'sqrt(mean(square(log(prediction)-log(published arithmetic mean)))) on positive mean positions',
    'gmfe':'exp(mean(abs(log(prediction)-log(published arithmetic mean)))) on the same positions',
    'fold_bias':'exp(mean(log(prediction)-log(published arithmetic mean))); above one denotes overprediction',
    'within_2fold_fraction':'mean(abs(log(prediction)-log(published arithmetic mean)) <= log(2)) on the same positions',
    'linear_errors':'MAE and RMSE in ng/mL over all available published mean positions, including source-reported zeros',
    'mean_curve_endpoints':{
        'domain':'entire fixed original nominal grid; unknown missing positions make both derived endpoints unavailable',
        'sampled_peak':'max of published zeroed mean curve and max of reference prediction on the same full grid',
        'observed_window_auc':'linear trapezoidal integral over the full published mean grid, with published zeros preserved as the reported target',
        'errors':'prediction/target ratio and absolute natural log ratio only for positive endpoint target and prediction',
        'qualification':'derived properties of the source-zeroed arithmetic mean curve; not mean individual Cmax or individual NCA AUC, not AUCinf'},
    'assay_mismatch':'unthresholded sex-weighted reference predictions cannot reproduce individual BLQ-zeroing before averaging without individual distributions; no invented correction',
    'missing_policy':'retain unknown missing positions with reason; report point and endpoint coverage without effect-based window selection',
    'failures':'all 57 method selections remain in denominators; missing sex or unsuccessful native prediction fails that model; no success-only seed mean',
    'statistical_unit':'one compound and one aggregate curve from one cohort/publication; cohort N is not an independent curve count',
    'uncertainty':'no bootstrap of invented individuals, time points or random seeds as independent drugs; no individual confidence interval reconstructed from group SD',
    'scientific_tier':'predeclared transportability sensitivity; successful execution cannot promote strict-main eligibility',
}


def local(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    if p==ROOT or not p.is_relative_to(ROOT):raise ValueError('Scoring resources must remain inside this project')
    return p


def sha(path):
    with local(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    path=local(path)
    if path.stat().st_size>64*1024**2:raise ValueError('Oversized scoring JSON')
    return json.loads(path.read_text())


def verified(path,checksum,*,outcome=False,denied=()):
    path=local(path)
    if not outcome and (path in denied or any(p.lower().startswith('sealed') for p in path.relative_to(ROOT).parts)):
        raise ValueError('Outcome/raw-source paths cannot enter the prediction-lock validation barrier')
    if not isinstance(checksum,str) or not re.fullmatch('[a-f0-9]{64}',checksum) or sha(path)!=checksum:
        raise ValueError('Hash mismatch: '+str(path.relative_to(ROOT)))
    return path


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def same_grid(actual,expected):
    return isinstance(actual,list) and len(actual)==len(expected) and all(
        finite(a) and math.isclose(a,b,rel_tol=0,abs_tol=1e-8) for a,b in zip(actual,expected))


def expected_selections():
    expected={}
    for block,(budget,dimension,methods) in BLOCKS.items():
        for method in methods:
            for seed in ((17,) if method in {'M0','M1','E0'} else (17,43,101)):
                key=f'{block}__{method}_seed_{seed}'
                expected[key]={'selection_id':key,'block_id':block,'method':method,'seed':seed,
                    'solver_budget':budget,'exact_free_parameter_count':dimension}
    return expected


def validate_selections(selections,models):
    expected=expected_selections()
    if not isinstance(selections,list) or len(selections)!=57:raise ValueError('All 57 method selections are required')
    by_id={s['selection_id']:s for s in selections}
    if len(by_id)!=57 or set(by_id)!=set(expected):raise ValueError('Method/seed denominator mismatch')
    keys=set()
    for key,wanted in expected.items():
        row=by_id[key]
        if any(row.get(k)!=v or type(row.get(k)) is not type(v) for k,v in wanted.items()):
            raise ValueError('Method, budget or fixed-parameter-count stratum changed')
        model=row.get('model_key')
        if model is None:
            if row.get('prediction_available') is not False or not row.get('failure'):
                raise ValueError('Unsubmitted models need an explicit failure reason')
        elif (not isinstance(model,str) or not model or row.get('prediction_available') is not True
            or row.get('upstream_status') not in {'completed','budget_exhausted'}):
            raise ValueError('Invalid available upstream model')
        else:keys.add(model)
    if not isinstance(models,dict) or set(models)!=keys:raise ValueError('Frozen model registry differs from nonfailed selections')


def validate_native_artifacts(row,protocol,times,denied):
    path=verified(row['native_csv_path'],row['native_csv_sha256'],denied=denied)
    column=protocol['native_concentration_column']
    if row.get('native_time_column')!='Time [min]' or row.get('native_concentration_column')!=column:
        raise ValueError('Native concentration/time column changed')
    if path.stat().st_size>128*1024**2:raise ValueError('Oversized native CSV')
    with path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    native_times=[float(r['Time [min]']) for r in rows];values=[float(r[column]) for r in rows]
    if (len(native_times)<2 or any(not finite(x) for x in native_times+values) or any(v<0 for v in values)
        or any(b<=a for a,b in zip(native_times,native_times[1:]))
        or times[0]<native_times[0]-1e-8 or times[-1]>native_times[-1]+1e-8):
        raise ValueError('Native curve must be finite, nonnegative and cover the unchanged nominal grid')
    sampled=[]
    for time in times:
        right=bisect.bisect_left(native_times,time)
        if right==0:value=values[0]
        elif right==len(native_times):value=values[-1]
        else:
            left=right-1;weight=(time-native_times[left])/(native_times[right]-native_times[left])
            value=values[left]+weight*(values[right]-values[left])
        sampled.append(value)
    if any(not math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10) for a,b in zip(sampled,row['concentration'])):
        raise ValueError('Locked nominal concentrations differ from native interpolation')
    archives=row.get('pkml_archives')
    if not isinstance(archives,list) or len(archives)!=1:raise ValueError('One restorable exact native model archive is required')
    for archive in archives:
        path=verified(archive['path'],archive['sha256'],denied=denied);local(archive['original_path'])
        size=archive['original_byte_size']
        if type(size) is not int or not 0<size<=256*1024**2:raise ValueError('Invalid bounded original PKML size')
        digest=hashlib.sha256();total=0
        with gzip.open(path,'rb') as stream:
            while chunk:=stream.read(1024**2):
                total+=len(chunk)
                if total>size:raise ValueError('Native archive exceeds registered size')
                digest.update(chunk)
        if total!=size or digest.hexdigest()!=archive['original_sha256']:
            raise ValueError('Archive cannot restore original native model bytes')


def validate_locked_inputs(lock_path):
    lock_path=local(lock_path)
    if any(p.lower().startswith('sealed') for p in lock_path.relative_to(ROOT).parts):raise ValueError('A lock cannot be an outcome file')
    lock=read(lock_path)
    if (lock.get('schema_version')!='pbpk-external-predictions-lock-v1' or lock.get('model_selection_frozen') is not True
        or lock.get('all_registered_predictions_terminal') is not True or lock.get('method_job_count')!=57
        or lock.get('expected_prediction_rows')!=57):raise ValueError('Complete 57-method aggregate prediction lock required')
    ref=lock['config_freeze'];config=read(verified(ref['path'],ref['sha256']))
    if config.get('evaluation_contract')!=EVALUATION_CONTRACT:raise ValueError('Aggregate evaluation must be registered before prediction')
    selections,models=config['selections'],config['models'];validate_selections(selections,models)
    records,cases=lock['predictions'],config['cases']
    if not isinstance(cases,list) or len(cases)!=1 or not isinstance(records,list) or len(records)!=1:
        raise ValueError('This lock evaluates exactly one registered drug/cohort aggregate curve')
    item,planned=records[0],cases[0]
    if any(item[k]!=planned[k] for k in ('case_id','design_path','design_sha256','protocol_path','protocol_sha256')):
        raise ValueError('Design/protocol changed after the global pre-run freeze')
    design=read(verified(item['design_path'],item['design_sha256']))
    excluded=design['evidence_exclusion']
    denied={local(design['sealed_observation_reference']['path']),
        *(local(p) for p in excluded.get('raw_source_files',[])),
        *(local(p) for p in excluded.get('curator_only_source_paths',[])),
        *(local(design[name]['path']) for name in ('source_article','source_pdf') if name in design)}
    protocol=read(verified(item['protocol_path'],item['protocol_sha256'],denied=denied))
    prediction=read(verified(item['path'],item['sha256'],denied=denied))
    if any(x.get('case_id')!=item['case_id'] for x in (design,protocol,prediction)):raise ValueError('Case identities differ')
    if (design.get('schema_version')!='pbpk-independent-aggregate-curve-design-v1'
        or design.get('primary_aggregation')!=AGGREGATION or design.get('aggregation_preselected_before_any_model_predictions') is not True
        or design.get('scientific_results_accessed_by_algorithm') is not False
        or design['censoring'].get('apply_individual_lloq_filter_to_positive_group_mean') is not False
        or design['evaluation_target'].get('individual_records_available') is not False
        or excluded.get('exclude_entire_publication_and_cohort_from_parameter_sources_prompts_retrieval_and_training') is not True
        or excluded.get('exclude_entire_trial') is not True):raise ValueError('Wrong aggregate estimand or source isolation contract')
    if (protocol.get('design_sha256')!=item['design_sha256'] or prediction.get('protocol_sha256')!=item['protocol_sha256']
        or protocol.get('evaluation_contract')!=EVALUATION_CONTRACT or protocol.get('primary_aggregation')!=AGGREGATION
        or protocol.get('prediction_aggregation')!='sex_weighted_reference_concentration'):
        raise ValueError('Prediction protocol is not bound to the frozen aggregate contract')
    if (protocol.get('selections')!=selections or prediction.get('selections')!=selections or protocol.get('models')!=models
        or set(protocol.get('model_keys',[]))!=set(models) or len(protocol.get('model_keys',[]))!=len(models)):
        raise ValueError('Final model/selection set differs from the pre-run freeze')
    qualification={k:config.get(k) for k in ('strict_main_eligible','qualification_tier','qualification_reasons')}
    if (qualification['strict_main_eligible'] is not False or qualification['qualification_tier']!='transportability_sensitivity'
        or not isinstance(qualification['qualification_reasons'],list) or not qualification['qualification_reasons']
        or any(not isinstance(s,str) or not s.strip() for s in qualification['qualification_reasons'])
        or any(protocol.get(k)!=v for k,v in qualification.items())):
        raise ValueError('Predeclared scientific qualification cannot be promoted or omitted')
    approximation=protocol.get('approximation',{})
    for key,value in {'reference_subjects_only':True,'group_mean_reference_mismatch':True,
        'source_individual_BLQ_zeroed_mean_vs_uncensored_prediction':True,'individual_covariates_available':False}.items():
        if approximation.get(key) is not value:raise ValueError('Reference-versus-source-mean approximation must be explicit')
    pop=design['population_metadata'];weights=protocol.get('sex_weights',{});mw=protocol.get('molecular_weight_g_mol')
    if (not finite(mw) or mw<=0 or set(weights)!={'male','female'} or type(pop['n']) is not int or pop['n']<=0
        or sum(pop['sex_counts'].values())!=pop['n'] or pop['n']!=design['baseline_curve_n']
        or any(not finite(w) or w<=0 for w in weights.values())
        or not math.isclose(sum(weights.values()),1,rel_tol=0,abs_tol=1e-12)
        or any(not math.isclose(weights[s],pop['sex_counts'][s]/pop['n'],rel_tol=0,abs_tol=1e-12) for s in weights)):
        raise ValueError('Weights must use the full baseline cohort, not the paired endpoint subset')
    times=[t*60 for t in design['sampling_times_h']]
    if (len(times)<2 or any(not finite(t) or t<0 for t in times) or any(b<=a for a,b in zip(times,times[1:]))
        or not same_grid(design['protocol']['sampling_times_min'],times)
        or not same_grid(protocol.get('sampling_time_min'),times)
        or len(times)!=design['evaluation_target']['scheduled_time_count']):raise ValueError('Original full nominal time grid changed')
    name=design['compound'].capitalize()
    expected_column=f'Organism|PeripheralVenousBlood|{name}|Plasma (Peripheral Venous Blood) [µmol/l]'
    if protocol.get('native_concentration_column')!=expected_column:raise ValueError('Native column must select the registered parent plasma output')
    rows=prediction.get('models');wanted={(k,s) for k in models for s in ('male','female')}
    if (not isinstance(rows,list) or len(rows)!=len(wanted)
        or {(r.get('model_key'),r.get('sex')) for r in rows}!=wanted):raise ValueError('Every selected model needs both sex terminal records')
    by_model={key:[] for key in models}
    for row in rows:
        status=row.get('status');values=row.get('concentration')
        if (status not in {'completed','native_failed','not_run_resource_limit'} or type(row.get('native_execution_invoked')) is not bool
            or status=='completed' and not row['native_execution_invoked'] or status=='not_run_resource_limit' and row['native_execution_invoked']
            or row.get('time_unit')!='min' or str(row.get('unit')).replace('µ','u').replace('μ','u')!='umol/l'
            or not same_grid(row.get('time_min'),times)):raise ValueError('Invalid native terminal status or sampling contract')
        if status=='completed':
            if (not isinstance(values,list) or len(values)!=len(times) or any(not finite(v) or v<0 for v in values)
                or row.get('error') is not None):raise ValueError('Completed sex curves must be finite and nonnegative')
            validate_native_artifacts(row,protocol,times,denied)
        elif values is not None or not row.get('error'):raise ValueError('Native failures must retain null predictions and explicit reasons')
        by_model[row['model_key']].append(row)
    jobs=config.get('expected_native_jobs');expected_jobs={(item['case_id'],k,s) for k,s in wanted}
    if (not isinstance(jobs,list) or len(jobs)!=len(expected_jobs)
        or {(j['case_id'],j['model_key'],j['sex']) for j in jobs}!=expected_jobs
        or lock.get('registered_native_jobs')!=len(expected_jobs)
        or lock.get('native_execution_attempts')!=sum(r['native_execution_invoked'] for r in rows)
        or lock.get('completed_native_jobs')!=sum(r['status']=='completed' for r in rows)):
        raise ValueError('Registered, attempted and completed native jobs must retain distinct correct counts')
    frozen=config.get('frozen_input_sha256')
    if not isinstance(frozen,dict) or not frozen:raise ValueError('Executed sources and inputs must be hashed before prediction')
    for path,checksum in frozen.items():verified(path,checksum,denied=denied)
    predictions={}
    for key,parts in by_model.items():
        failed=[{'sex':r['sex'],'status':r['status'],'error':r.get('error')} for r in parts if r['status']!='completed']
        predictions[key]={'native_failures':failed,'concentration_ng_ml':None if failed else
            [mw*sum(weights[r['sex']]*r['concentration'][i] for r in parts) for i in range(len(times))]}
    return lock,config,[{'case_id':item['case_id'],'design':design,'protocol':protocol,'time_min':times,
        'models':predictions,'qualification':qualification}]


def read_outcomes(case):
    """First outcome access: caller must have validated the entire lock."""
    design=case['design'];ref=design['sealed_observation_reference'];sealed=read(verified(ref['path'],ref['sha256'],outcome=True))
    if (sealed.get('schema_version')!='pbpk-independent-aggregate-curves-sealed-v1'
        or any(sealed.get(k)!=design[k] for k in ('case_id','compound','publication_cluster_id','cohort_id','trial_cluster_ids'))
        or sealed.get('primary_aggregation')!=AGGREGATION or sealed.get('individual_records_available') is not False
        or sealed.get('population_n')!=design['population_metadata']['n'] or sealed.get('concentration_unit')!='ng/mL'
        or sealed.get('time_unit')!='h' or sealed.get('censoring')!=design['censoring']):raise ValueError('Sealed aggregate source contract mismatch')
    rows=sealed['records']
    if len(rows)!=len(case['time_min']):raise ValueError('Every original source row must be retained')
    for index,(row,time) in enumerate(zip(rows,case['time_min'])):
        if row.get('source_row_ordinal')!=index+1 or not finite(row.get('time_h')) or abs(row['time_h']*60-time)>1e-8:
            raise ValueError('Sealed row order or nominal grid changed')
        value=row.get('arithmetic_mean_ng_ml')
        if value is not None and (not finite(value) or value<0):raise ValueError('Invalid published arithmetic mean')
        if value is None and row.get('arithmetic_mean_missing_reason')!='source_missing_unspecified':
            raise ValueError('Missing published mean needs an explicit source reason')
        if value==0 and (row.get('published_arithmetic_mean_is_zero') is not True or not row.get('zero_interpretation')):
            raise ValueError('Published zero must retain its source interpretation')
    return rows


def curve_metrics(records,times,predicted):
    points=[];log_errors=[];linear=[]
    if len(records)!=len(times) or len(predicted)!=len(times) or any(not finite(v) or v<0 for v in predicted):
        raise ValueError('Predictions must be finite and nonnegative on the complete original grid')
    invalid=False
    for i,(row,pred) in enumerate(zip(records,predicted)):
        observed=row['arithmetic_mean_ng_ml'];positive=observed is not None and observed>0
        reason='source_missing_unspecified' if observed is None else 'source_zeroed_or_rounded_mean_log_undefined' if observed==0 else None
        points.append({'source_row_ordinal':row['source_row_ordinal'],'time_min':times[i],
            'arithmetic_mean_ng_ml':observed,'prediction_ng_ml':pred,'included_in_log_error':positive,
            'log_exclusion_reason':reason,'source_zero_interpretation':row.get('zero_interpretation')})
        if observed is not None:linear.append(pred-observed)
        if positive:
            if pred<=0:invalid=True
            else:log_errors.append(math.log(pred)-math.log(observed))
    complete=not invalid and bool(log_errors)
    result={'status':'evaluated' if complete else 'failed_retained_in_denominator',
        'failure':'nonpositive_prediction_for_positive_group_mean' if invalid else None if log_errors else 'no_positive_published_mean',
        'points':points,'original_points_expected':len(times),'published_values_available':len(linear),
        'positive_mean_points':sum(p['included_in_log_error'] for p in points),
        'point_coverage':len(linear)/len(times),'log_rmse':None,'gmfe':None,'fold_bias':None,'within_2fold_fraction':None,
        'mean_log_ratio':None,'mean_absolute_log_ratio':None,
        'mae_ng_ml':sum(abs(v) for v in linear)/len(linear) if linear else None,
        'rmse_ng_ml':math.sqrt(sum(v*v for v in linear)/len(linear)) if linear else None,
        'individual_lloq_filter_applied':False,'point_imputation':False}
    if complete:
        signed=sum(log_errors)/len(log_errors);absolute=sum(abs(v) for v in log_errors)/len(log_errors)
        result.update(log_rmse=math.sqrt(sum(v*v for v in log_errors)/len(log_errors)),
            mean_log_ratio=signed,mean_absolute_log_ratio=absolute,gmfe=math.exp(absolute),fold_bias=math.exp(signed),
            within_2fold_fraction=sum(.5*p['arithmetic_mean_ng_ml']<=p['prediction_ng_ml']<=2*p['arithmetic_mean_ng_ml']
                for p in points if p['included_in_log_error'])/len(log_errors))
    endpoints={'domain_complete':len(linear)==len(times),'grid_coverage':len(linear)/len(times),
        'window_min':[times[0],times[-1]],'sampled_peak_of_reported_mean':None,'observed_window_auc_of_reported_mean':None,
        'reason':None if len(linear)==len(times) else 'unknown_missing_mean_positions',
        'not_mean_individual_NCA':True,'source_zeroed_mean_not_reconstructed_uncensored_curve':True}
    if endpoints['domain_complete']:
        observed=[r['arithmetic_mean_ng_ml'] for r in records]
        def auc(values):return sum((b-a)*(x+y)/2 for a,b,x,y in zip(times,times[1:],values,values[1:]))
        for name,obs,pred,unit in [('sampled_peak_of_reported_mean',max(observed),max(predicted),'ng/mL'),
            ('observed_window_auc_of_reported_mean',auc(observed),auc(predicted),'ng*min/mL')]:
            endpoints[name]={'reported_mean_target':obs,'reference_prediction':pred,'unit':unit,
                'ratio':pred/obs if obs>0 else None,'absolute_log_error':abs(math.log(pred)-math.log(obs)) if obs>0 and pred>0 else None,
                'error_evaluable':obs>0 and pred>0,
                'reason':None if obs>0 and pred>0 else 'nonpositive_endpoint_has_no_log_ratio'}
    result['mean_curve_endpoints']=endpoints
    return result


def score_locked(lock_path,output):
    lock_path,output=local(lock_path),local(output)
    if output.exists():raise ValueError('Scoring output is immutable; use a new registered output')
    lock,config,cases=validate_locked_inputs(lock_path)
    # GLOBAL BARRIER: all 57 choices, both sex records, code/input/CSV/model hashes
    # and the original grid passed before the first sealed observation is opened.
    case=cases[0];records=read_outcomes(case);unique={}
    for key,prediction in case['models'].items():
        if prediction['native_failures']:
            unique[key]={'model_key':key,'status':'failed_retained_in_denominator','failure':'native_prediction_failed',
                'native_failures':prediction['native_failures'],'log_rmse':None,'gmfe':None,'fold_bias':None,
                'within_2fold_fraction':None,'mean_log_ratio':None,'mean_absolute_log_ratio':None,
                'mae_ng_ml':None,'rmse_ng_ml':None,'mean_curve_endpoints':None}
        else:unique[key]={'model_key':key,**curve_metrics(records,case['time_min'],prediction['concentration_ng_ml'])}
    method_rows=[]
    for selection in config['selections']:
        scored=unique.get(selection['model_key'])
        method_rows.append({**selection,'case_id':case['case_id'],'compound':case['design']['compound'],
            'publication_cluster_id':case['design']['publication_cluster_id'],'trial_cluster_ids':case['design']['trial_cluster_ids'],
            **case['qualification'],'status':scored['status'] if scored else 'failed_retained_in_denominator',
            'failure':scored.get('failure') if scored else selection.get('failure'),
            **{k:scored.get(k) if scored else None for k in ('log_rmse','gmfe','fold_bias','within_2fold_fraction',
                'mean_log_ratio','mean_absolute_log_ratio','mae_ng_ml','rmse_ng_ml','mean_curve_endpoints')}})
    strata=[]
    for block,(budget,dimension,methods) in BLOCKS.items():
        for method in methods:
            rows=[r for r in method_rows if r['block_id']==block and r['method']==method]
            complete=all(r['status']=='evaluated' for r in rows)
            strata.append({'block_id':block,'method':method,'solver_budget':budget,'exact_free_parameter_count':dimension,
                'seeds_expected':[r['seed'] for r in rows],'seeds_evaluated':sum(r['status']=='evaluated' for r in rows),
                'mean_log_rmse':sum(r['log_rmse'] for r in rows)/len(rows) if complete else None,
                'gmfe':math.exp(sum(r['mean_absolute_log_ratio'] for r in rows)/len(rows)) if complete else None,
                'fold_bias':math.exp(sum(r['mean_log_ratio'] for r in rows)/len(rows)) if complete else None,
                'within_2fold_fraction':sum(r['within_2fold_fraction'] for r in rows)/len(rows) if complete else None,
                'mean_curve_endpoints_by_seed':{str(r['seed']):r['mean_curve_endpoints'] for r in rows},
                'success_only_averaging':False,'confidence_interval':None,
                'CI_reason':'One published group-mean curve; cohort participants, sampling times and seeds are not independent drug observations'})
    output.mkdir(parents=True);sealed_dir=output/'sealed';sealed_dir.mkdir(mode=0o700)
    score_path=sealed_dir/'aggregate_curve_scores.json'
    scores={'schema_version':'pbpk-all-method-aggregate-curve-scores-v1','case_id':case['case_id'],
        'unique_models':list(unique.values()),'method_scenarios':method_rows,'method_budget_strata':strata,
        'evaluation_contract':EVALUATION_CONTRACT,'primary_aggregation':AGGREGATION,'prediction_approximation':case['protocol']['approximation'],
        **case['qualification'],'independent_drugs':1,'independent_studies':1,'aggregate_curves':1,
        'source_cohort_n':case['design']['population_metadata']['n'],'independent_individual_curves':0,
        'publication_cluster_id':case['design']['publication_cluster_id'],'trial_cluster_ids':case['design']['trial_cluster_ids'],
        'method_jobs_expected':57,'method_scenario_rows_expected':57,'prediction_lock_sha256':sha(lock_path),
        'scores_sealed_from_method_development':True}
    score_path.write_text(json.dumps(scores,indent=2,allow_nan=False)+'\n');score_path.chmod(0o600)
    receipt={'status':'scored_and_sealed','method_jobs_expected':57,'method_scenario_rows_expected':57,
        'method_scenario_rows_evaluated':sum(r['status']=='evaluated' for r in method_rows),
        'method_scenario_rows_failed_retained':sum(r['status']!='evaluated' for r in method_rows),
        'independent_drugs':1,'independent_studies':1,'aggregate_curves':1,'independent_individual_curves':0,
        'source_cohort_n':case['design']['population_metadata']['n'],**case['qualification'],
        'publication_cluster_id':case['design']['publication_cluster_id'],'trial_cluster_ids':case['design']['trial_cluster_ids'],
        'primary_aggregation':AGGREGATION,'prediction_approximation':case['protocol']['approximation'],
        'sealed_scores_path':str(score_path.relative_to(ROOT)),'sealed_scores_sha256':sha(score_path),
        'lock_path':str(lock_path.relative_to(ROOT)),'lock_sha256':sha(lock_path),'scorer_sha256':sha(Path(__file__)),
        'metrics_exposed_to_method_development':False}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--lock',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    if args.validate_only:
        _,config,cases=validate_locked_inputs(local(args.lock))
        print(json.dumps({'status':'lock_validated','method_jobs':len(config['selections']),'aggregate_curves':len(cases),'outcomes_read':False}));return
    print(json.dumps(score_locked(local(args.lock),local(args.output)),indent=2))


if __name__=='__main__':main()
