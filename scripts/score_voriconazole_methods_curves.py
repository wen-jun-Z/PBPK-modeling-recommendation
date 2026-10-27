#!/usr/bin/env python3
"""Score every frozen Vori method choice; expose execution counts, never rankings.

The complete lock is validated before the first access to sealed observations.
Two dose cohorts are one drug/publication, not 114 independent experiments.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from score_locked_independent_curves import finite_number,score_subject

BLOCKS={
    'main_5000':(5000,None,('M0','M1','M2','M3','M4','M5','M6','A1','A2','A3','E0')),
    'budget_500':(500,None,('M3','M4','M5','M6')),
    'budget_2000':(2000,None,('M3','M4','M5','M6')),
    'matched_1':(5000,1,('M5','M6')),
}
TERMINAL={'completed','native_failed','not_run_resource_limit'}
TARGET='Organism|PeripheralVenousBlood|Voriconazole1|Plasma (Peripheral Venous Blood) [µmol/l]'
MAX_JSON_BYTES=64*1024**2
EVALUATION_CONTRACT={
    'version':'vori-all-method-curve-evaluation-v1',
    'primary':'arithmetic mean of subject natural-log RMSE, then equal dose-cohort mean',
    'point_domain':'source quantified observations strictly positive and at least source LLOQ; no imputation',
    'auxiliary_same_concentration_error_dimension':{
        'gmfe':'exp(mean absolute natural log(prediction/observation))',
        'fold_bias':'exp(mean natural log(prediction/observation)); above one denotes overprediction',
        'within_2fold_fraction':'mean(abs(log(prediction/observation)) <= log(2))'},
    'auxiliary_aggregation':'equal subjects within dose, equal dose cohorts within study; average log components before exp for GMFE and fold bias',
    'sampled_cmax_and_observed_window_auc':{
        'window':'entire preregistered original nominal sampling grid from its first to last time',
        'eligibility':'every grid position quantified, strictly positive and at least source LLOQ; no BQL or missing positions',
        'cmax':'maximum concentration at the registered samples; not continuous-time clinical Cmax',
        'auc':'linear trapezoidal integral over original sampling times, micromolar*minute; not AUCinf',
        'error':'absolute natural log(predicted endpoint/observed endpoint)',
        'coverage':'eligible complete subject grids / all registered reference subject grids, by dose and study',
        'summary':'conditional mean endpoint absolute log error among complete grids, always accompanied by coverage; study summary requires coverage in both dose cohorts'},
    'censored_pk_auxiliary':{
        'domain':'same entire original nominal grid; unspecified missing positions make both endpoints unavailable',
        'censoring_bounds':'source BQL, or a quantified value below assay LLOQ, contributes [0,LLOQ] only as lower/upper bounds, never a point imputation',
        'sampled_cmax':'exact sampled observed peak when at least one quantified peak is >=LLOQ and exceeds all censored upper bounds; predicted peak is max over the full original grid',
        'auc':'linear trapezoidal lower/upper integrals; exact quantifiable positions have identical lower and upper values',
        'auc_error':'absolute distance of predicted AUC to observed interval, plus distance divided by observed upper bound',
        'auc_interval_warning':'falling inside a wide interval establishes interval compatibility only, not precise predictive accuracy',
        'coverage':'all registered subject grids remain the denominator; report eligibility and interval widths by cohort'},
    'failed_predictions':'retained in all method denominators; no success-only primary mean, clipping or epsilon',
    'independent_unit':'one drug/publication; doses, subjects and random seeds are not independent drugs',
}


def local(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    if p==ROOT or not p.is_relative_to(ROOT):raise ValueError('All scoring resources must remain in this project')
    return p


def prebarrier_path(value):
    p=local(value)
    if any(part.lower().startswith('sealed') for part in p.relative_to(ROOT).parts):
        raise ValueError('An outcome path cannot enter the pre-observation lock barrier')
    return p


def sha(path):
    path=local(path)
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    path=local(path)
    if path.stat().st_size>MAX_JSON_BYTES:raise ValueError('Oversized scoring metadata')
    return json.loads(path.read_text())


def verified(value,expected,*,sealed=False):
    path=local(value) if sealed else prebarrier_path(value)
    if not isinstance(expected,str) or not re.fullmatch('[a-f0-9]{64}',expected) or sha(path)!=expected:
        raise ValueError('Hash mismatch: '+str(path.relative_to(ROOT)))
    return path


def expected_selections():
    result={}
    for block,(budget,dimension,methods) in BLOCKS.items():
        for method in methods:
            for seed in ((17,) if method in {'M0','M1','E0'} else (17,43,101)):
                identity=f'{block}__{method}_seed_{seed}'
                result[identity]={'selection_id':identity,'block_id':block,'method':method,'seed':seed,
                    'solver_budget':budget,'exact_free_parameter_count':dimension}
    return result


def validate_selections(selections,models):
    expected=expected_selections()
    if not isinstance(selections,list) or len(selections)!=57:
        raise ValueError('The full 57 method-job denominator is required')
    by_id={s.get('selection_id'):s for s in selections}
    if len(by_id)!=57 or set(by_id)!=set(expected):raise ValueError('Selection identities differ from all registered blocks/seeds')
    keys=set()
    for key,wanted in expected.items():
        selected=by_id[key]
        if any(selected.get(k)!=v or type(selected.get(k)) is not type(v) for k,v in wanted.items()):
            raise ValueError('A method/budget/dimension stratum was changed')
        model=selected.get('model_key')
        if model is None:
            if selected.get('prediction_available') is not False or not selected.get('failure'):
                raise ValueError('Failed upstream selections need an explicit retained reason')
        else:
            if (not isinstance(model,str) or not model or selected.get('prediction_available') is not True
                or selected.get('upstream_status') not in {'completed','budget_exhausted'}):
                raise ValueError('Invalid available upstream selection')
            keys.add(model)
    if not isinstance(models,dict) or set(models)!=keys:
        raise ValueError('Registered models must exactly cover the nonfailed method selections')
    return by_id


def same_grid(actual,expected):
    return (isinstance(actual,list) and len(actual)==len(expected) and
        all(finite_number(a) and math.isclose(a,b,rel_tol=0,abs_tol=1e-8) for a,b in zip(actual,expected)))


def validate_csv(model,expected_times):
    """The nominal concentrations must come from the retained native CSV."""
    path=verified(model['native_csv_path'],model['native_csv_sha256'])
    if path.stat().st_size>128*1024**2:raise ValueError('Oversized native curve CSV')
    with path.open(encoding='utf-8-sig',newline='') as stream:
        records=list(csv.DictReader(stream))
    times=[float(row['Time [min]']) for row in records]
    values=[float(row[TARGET]) for row in records]
    if (len(times)<2 or any(not finite_number(x) for x in times+values) or any(v<0 for v in values)
        or any(b<=a for a,b in zip(times,times[1:]))
        or expected_times[0]<times[0]-1e-8 or expected_times[-1]>times[-1]+1e-8):
        raise ValueError('Native curve is nonfinite, unordered or does not cover the nominal grid')
    import numpy as np
    sampled=np.interp(expected_times,times,values).tolist()
    if any(not math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10) for a,b in zip(sampled,model['concentration'],strict=True)):
        raise ValueError('Locked nominal prediction differs from native CSV interpolation')


def validate_archives(model):
    archives=model.get('pkml_archives')
    if not isinstance(archives,list) or not archives:
        raise ValueError('Completed predictions must retain restorable native PKML archives')
    if len({r['path'] for r in archives})!=len(archives):raise ValueError('Duplicate native archive')
    for record in archives:
        path=verified(record['path'],record['sha256']);prebarrier_path(record['original_path'])
        size=record.get('original_byte_size')
        if type(size) is not int or not 0<size<=256*1024**2:
            raise ValueError('Invalid bounded original PKML size')
        digest=hashlib.sha256();total=0
        with gzip.open(path,'rb') as stream:
            while chunk:=stream.read(1024**2):
                total+=len(chunk)
                if total>size:raise ValueError('Archive exceeds its frozen decompressed size')
                digest.update(chunk)
        if total!=size or digest.hexdigest()!=record['original_sha256']:
            raise ValueError('Native archive does not reproduce the exact generated PKML')


def validate_locked_inputs(lock_path):
    lock_path=prebarrier_path(lock_path);lock=read(lock_path)
    if (lock.get('schema_version')!='pbpk-external-predictions-lock-v1'
        or lock.get('model_selection_frozen') is not True
        or lock.get('all_registered_predictions_terminal') is not True
        or lock.get('method_job_count')!=57 or lock.get('expected_prediction_rows')!=114):
        raise ValueError('An explicit complete all-method prediction lock is required')
    ref=lock['config_freeze'];config=read(verified(ref['path'],ref['sha256']))
    if config.get('evaluation_contract')!=EVALUATION_CONTRACT:
        raise ValueError('All primary/auxiliary metric definitions must be frozen before prediction')
    selections=config['selections'];models=config['models']
    validate_selections(selections,models)
    cases=config['cases'];records=lock['predictions']
    if (not isinstance(cases,list) or len(cases)!=2 or len(records)!=2
        or len({c['case_id'] for c in cases})!=2 or len({c['case_id'] for c in records})!=2
        or {c['case_id'] for c in cases}!={c['case_id'] for c in records}):
        raise ValueError('Both preregistered dose cases must be retained exactly once')
    frozen={c['case_id']:c for c in cases};checked=[];denied=set();native_status_rows=[]
    for item in records:
        case_id=item['case_id'];planned=frozen[case_id]
        if any(item[k]!=planned[k] for k in ('design_path','design_sha256','protocol_path','protocol_sha256')):
            raise ValueError('A design/protocol changed after the pre-run freeze')
        design=read(verified(item['design_path'],item['design_sha256']))
        protocol=read(verified(item['protocol_path'],item['protocol_sha256']))
        prediction=read(verified(item['path'],item['sha256']))
        denied.add(local(design['sealed_observation_reference']['path']))
        if any(x.get('case_id')!=case_id for x in (design,protocol,prediction)):
            raise ValueError('Case identity mismatch')
        if (design.get('schema_version')!='pbpk-independent-curve-design-v1'
            or protocol.get('design_sha256')!=item['design_sha256']
            or prediction.get('protocol_sha256')!=item['protocol_sha256']):
            raise ValueError('Predictions must bind the exact registered design and protocol')
        if (protocol.get('selections')!=selections or prediction.get('selections')!=selections
            or protocol.get('models')!=models or set(protocol.get('model_keys',[]))!=set(models)
            or len(protocol.get('model_keys',[]))!=len(models)):
            raise ValueError('Predicted models/selections differ from the pre-run freeze')
        if protocol.get('evaluation_contract')!=EVALUATION_CONTRACT:
            raise ValueError('Protocol metric definitions differ from the pre-run contract')
        if protocol.get('prediction_aggregation')!='sex_weighted_reference_concentration':
            raise ValueError('Unregistered demographic prediction aggregation')
        lloq=design['curve_evaluation_contract']['assay_LLOQ_ng_ml']
        if not finite_number(lloq) or lloq<=0:raise ValueError('A positive source assay limit is required')
        mw=protocol.get('molecular_weight_g_mol');weights=protocol.get('sex_weights',{})
        if not finite_number(mw) or mw<=0 or set(weights)!={'male','female'}:
            raise ValueError('Positive molecular weight and explicit sex weights required')
        pop=design['population']
        if (any(not finite_number(v) or v<=0 for v in weights.values())
            or not math.isclose(sum(weights.values()),1,abs_tol=1e-12)
            or any(not math.isclose(weights[s],pop[s+'_n']/pop['n'],abs_tol=1e-12) for s in weights)):
            raise ValueError('Demographic weights differ from the registered cohort')
        expected_times=[t*60 for t in design['sampling']['time_h']]
        if (not expected_times or any(not finite_number(t) or t<0 for t in expected_times)
            or any(b<=a for a,b in zip(expected_times,expected_times[1:]))
            or not same_grid(protocol.get('sampling_time_min'),expected_times)):
            raise ValueError('Original nominal sampling schedule must be retained')
        rows=prediction.get('models');wanted={(k,s) for k in models for s in weights}
        if (not isinstance(rows,list) or len(rows)!=len(wanted)
            or {(r.get('model_key'),str(r.get('sex')).lower()) for r in rows}!=wanted):
            raise ValueError('Each model needs both sex outcomes, including failures')
        by_model={k:[] for k in models}
        for row in rows:
            if (row.get('status') not in TERMINAL or row.get('time_unit')!='min'
                or str(row.get('unit')).replace('µ','u').replace('μ','u')!='umol/l'
                or not same_grid(row.get('time_min'),expected_times)):
                raise ValueError('Invalid model terminal state, unit or shifted sampling grid')
            if (type(row.get('native_execution_invoked')) is not bool
                or row['status']=='completed' and row['native_execution_invoked'] is not True
                or row['status']=='not_run_resource_limit' and row['native_execution_invoked'] is not False):
                raise ValueError('Executed and unexecuted native jobs must be distinguished')
            values=row.get('concentration')
            if row['status']=='completed':
                if (not isinstance(values,list) or len(values)!=len(expected_times)
                    or any(not finite_number(v) or v<0 for v in values) or row.get('error') is not None):
                    raise ValueError('A completed native prediction needs finite concentrations and no error')
                validate_csv(row,expected_times)
                validate_archives(row)
            elif values is not None or not row.get('error'):
                raise ValueError('Native failures need null predictions and explicit reasons')
            by_model[row['model_key']].append(row)
            native_status_rows.append(row)
        prepared={}
        for key,rows in by_model.items():
            failures=[{'sex':r['sex'],'status':r['status'],'error':r.get('error')} for r in rows if r['status']!='completed']
            prepared[key]={'status':'failed_retained_in_denominator' if failures else 'available',
                'native_failures':failures,'concentration_umol_l':None if failures else
                [sum(weights[str(r['sex']).lower()]*r['concentration'][i] for r in rows) for i in range(len(expected_times))]}
        checked.append({'case_id':case_id,'design':design,'protocol':protocol,'time_min':expected_times,'models':prepared})
    # Never open a source list until BOTH protocol-defined outcome paths are known.
    inputs=config.get('frozen_input_sha256')
    if not isinstance(inputs,dict) or not inputs:raise ValueError('Frozen executed sources and scientific inputs are required')
    for path,checksum in inputs.items():
        if local(path) in denied:raise ValueError('Sealed observations cannot be registered as prediction inputs')
        verified(path,checksum)
    expected_jobs={(c['case_id'],k,s) for c in checked for k in models for s in ('male','female')}
    jobs=config.get('expected_native_jobs')
    if (not isinstance(jobs,list) or len(jobs)!=len(expected_jobs)
        or {(r['case_id'],r['model_key'],str(r['sex']).lower()) for r in jobs}!=expected_jobs):
        raise ValueError('Complete case/model/sex native-job registration required')
    if (lock.get('registered_native_jobs')!=len(expected_jobs)
        or lock.get('native_execution_attempts')!=sum(r['native_execution_invoked'] for r in native_status_rows)
        or lock.get('completed_native_jobs')!=sum(r['status']=='completed' for r in native_status_rows)):
        raise ValueError('Native execution counters differ from the complete terminal job records')
    if (len({c['design']['compound'] for c in checked})!=1
        or len({c['design']['publication_cluster_id'] for c in checked})!=1):
        raise ValueError('This frozen evaluation is exactly one drug/publication, with two cohorts')
    return lock,config,checked


def reference_subjects(case):
    """Called exclusively after all prediction/input checks have passed."""
    design=case['design'];ref=design['sealed_observation_reference']
    sealed=read(verified(ref['path'],ref['sha256'],sealed=True))
    if (sealed.get('case_id')!=case['case_id'] or sealed.get('publication_cluster_id')!=design['publication_cluster_id']
        or sealed.get('unit')!='ng/ml'
        or sealed.get('assay_LLOQ_ng_ml')!=design['curve_evaluation_contract']['assay_LLOQ_ng_ml']):
        raise ValueError('Sealed source measurement contract mismatch')
    rows=[r for r in sealed['records'] if r['treatment']=='R']
    links={r['subject_id']:r for r in design['crossover']['all_subject_links']}
    if len(links)!=design['population']['n'] or len(rows)!=len(links) or {r['subject_id'] for r in rows}!=set(links):
        raise ValueError('All registered reference subjects must be retained')
    for row in rows:
        link=links[row['subject_id']]
        if row['period']!=link['periods']['R'] or row['sequence']!=link['sequence']:
            raise ValueError('Reference-treatment crossover linkage changed')
    return rows,sealed['assay_LLOQ_ng_ml']


def score_subject_metrics(record,times,predicted,mw,lloq):
    result=score_subject(record,times,predicted,mw,lloq)
    result.update(mean_log_ratio=None,mean_absolute_log_ratio=None,gmfe=None,fold_bias=None,within_2fold_fraction=None)
    if result['status']=='evaluated':
        ratios=[math.log(predicted[p['source_index']]/(record['observations'][p['source_index']]['concentration_ng_ml']/mw))
            for p in result['points'] if p['included']]
        signed=sum(ratios)/len(ratios);absolute=sum(abs(v) for v in ratios)/len(ratios)
        result.update(mean_log_ratio=signed,mean_absolute_log_ratio=absolute,gmfe=math.exp(absolute),
            fold_bias=math.exp(signed),within_2fold_fraction=sum(abs(v)<=math.log(2) for v in ratios)/len(ratios))
    missing=[{'source_index':p['source_index'],'reason':p['exclusion_reason']} for p in result['points'] if not p['included']]
    complete=result['status']=='evaluated' and len(times)>=2 and not missing
    endpoint={'evaluable':complete,'window_min':[times[0],times[-1]],'missing_domain':missing,
        'reason':None if complete else ('incomplete_quantified_original_grid' if missing else 'invalid_prediction_or_grid'),
        'sampled_cmax':None,'observed_window_auc':None}
    if complete:
        observed=[p['concentration_ng_ml']/mw for p in record['observations']]
        def auc(values):return sum((b-a)*(x+y)/2 for a,b,x,y in zip(times,times[1:],values,values[1:]))
        for name,obs,pred,unit in [('sampled_cmax',max(observed),max(predicted),'µmol/l'),
            ('observed_window_auc',auc(observed),auc(predicted),'µmol*min/l')]:
            endpoint[name]={'observed':obs,'predicted':pred,'unit':unit,'fold_ratio':pred/obs,
                'absolute_log_error':abs(math.log(pred/obs))}
    result['strict_observed_window_endpoints']=endpoint
    result['censored_pk_endpoints']=censored_pk_endpoints(record,times,predicted,mw,lloq)
    return result


def censored_pk_endpoints(record,times,predicted,mw,lloq):
    result={'sampled_cmax':None,'observed_window_auc_interval':None,'reason':None,
        'window_min':[times[0],times[-1]],'point_imputation':False,'prediction_clipping':False}
    if (len(times)<2 or len(predicted)!=len(times) or len(record['observations'])!=len(times)
        or any(not finite_number(v) or v<0 for v in predicted)):
        result['reason']='invalid_native_prediction_or_grid';return result
    lower=[];upper=[];quantified=[];censored=[];missing=[]
    for index,point in enumerate(record['observations']):
        status=point['observation_status'];value=point['concentration_ng_ml']
        if status=='source_missing_unspecified':missing.append(index);continue
        if status=='left_censored_below_LLOQ':
            lo,hi=0.,lloq/mw;censored.append(index)
        elif status=='quantified' and finite_number(value) and value>=0:
            if value<lloq:lo,hi=0.,lloq/mw;censored.append(index)
            else:lo=hi=value/mw;quantified.append(lo)
        else:raise ValueError('Invalid source observation for censored PK bounds')
        lower.append(lo);upper.append(hi)
    if missing:
        result.update(reason='unspecified_missing_positions_no_finite_source_bounds',missing_positions=missing);return result
    if quantified and max(quantified)>=max(upper):
        observed=max(quantified);prediction=max(predicted)
        result['sampled_cmax']={'observed':observed,'predicted':prediction,'unit':'µmol/l',
            'fold_ratio':prediction/observed,'absolute_log_error':abs(math.log(prediction/observed)) if prediction>0 else None,
            'reason':None if prediction>0 else 'nonpositive_prediction_at_observed_peak',
            'exact_sampled_peak_despite_other_left_censored_positions':bool(censored)}
    else:result['sampled_cmax_unavailable_reason']='all_positions_censored_no_exact_quantified_peak'
    def auc(values):return sum((b-a)*(x+y)/2 for a,b,x,y in zip(times,times[1:],values,values[1:]))
    lo,hi,pred=auc(lower),auc(upper),auc(predicted)
    distance=max(lo-pred,pred-hi,0.)
    result['observed_window_auc_interval']={'observed_lower':lo,'observed_upper':hi,'observed_interval_width':hi-lo,
        'predicted':pred,'unit':'µmol*min/l','censored_positions':censored,'exact_point_auc':not censored,
        'absolute_distance_to_interval':distance,'distance_over_observed_upper_bound':distance/hi if hi>0 else None,
        'within_observed_interval':lo<=pred<=hi,'within_interval_is_not_precise_accuracy':True}
    return result


def summarize_subject_metrics(scored,expected):
    complete=len(scored)==expected and all(s['status']=='evaluated' for s in scored)
    result={'gmfe':None,'fold_bias':None,'within_2fold_fraction':None,
        'mean_log_ratio':None,'mean_absolute_log_ratio':None}
    if complete:
        signed=sum(s['mean_log_ratio'] for s in scored)/expected
        absolute=sum(s['mean_absolute_log_ratio'] for s in scored)/expected
        result.update(gmfe=math.exp(absolute),fold_bias=math.exp(signed),mean_log_ratio=signed,
            mean_absolute_log_ratio=absolute,within_2fold_fraction=sum(s['within_2fold_fraction'] for s in scored)/expected)
    eligible=[s for s in scored if s['strict_observed_window_endpoints']['evaluable']]
    result['strict_endpoint_summary']={'subjects_expected':expected,'subjects_evaluable':len(eligible),
        'coverage':len(eligible)/expected,'conditional_on_complete_original_grid':True,
        **{name+'_mean_absolute_log_error':sum(s['strict_observed_window_endpoints'][name]['absolute_log_error'] for s in eligible)/len(eligible)
           if eligible else None for name in ('sampled_cmax','observed_window_auc')}}
    cmax=[s['censored_pk_endpoints']['sampled_cmax'] for s in scored
        if s['censored_pk_endpoints']['sampled_cmax'] is not None and s['censored_pk_endpoints']['sampled_cmax']['absolute_log_error'] is not None]
    auc=[s['censored_pk_endpoints']['observed_window_auc_interval'] for s in scored
        if s['censored_pk_endpoints']['observed_window_auc_interval'] is not None]
    result['censored_pk_summary']={'subjects_expected':expected,'sampled_cmax_evaluable':len(cmax),
        'sampled_cmax_coverage':len(cmax)/expected,
        'sampled_cmax_mean_absolute_log_error':sum(v['absolute_log_error'] for v in cmax)/len(cmax) if cmax else None,
        'auc_interval_evaluable':len(auc),'auc_interval_coverage':len(auc)/expected,
        'auc_mean_distance_over_upper_bound':sum(v['distance_over_observed_upper_bound'] for v in auc)/len(auc) if auc and all(v['distance_over_observed_upper_bound'] is not None for v in auc) else None,
        'auc_mean_interval_width':sum(v['observed_interval_width'] for v in auc)/len(auc) if auc else None,
        'auc_unit':'µmol*min/l','conditional_on_endpoint_eligibility':True,'interval_compatibility_is_not_point_accuracy':True}
    return result


def score_locked(lock_path,output):
    lock_path=local(lock_path);output=local(output)
    if output.exists():raise ValueError('Scoring output must be new; no result overwrite')
    lock,config,checked=validate_locked_inputs(lock_path)
    # Strict barrier: this is the first function allowed to open outcome files.
    outcomes={c['case_id']:reference_subjects(c) for c in checked}
    unique_cases={};method_cases=[];method_studies=[]
    for case in checked:
        subjects,lloq=outcomes[case['case_id']]
        for model,prediction in case['models'].items():
            scored=[]
            if prediction['status']=='available':
                scored=[score_subject_metrics(row,case['time_min'],prediction['concentration_umol_l'],
                    case['protocol']['molecular_weight_g_mol'],lloq) for row in subjects]
            complete=len(scored)==len(subjects) and all(r['status']=='evaluated' for r in scored)
            unique_cases[(case['case_id'],model)]={'case_id':case['case_id'],'model_key':model,
                'status':'evaluated' if complete else 'failed_retained_in_denominator',
                'reference_subjects_expected':len(subjects),'subjects_evaluated':sum(r['status']=='evaluated' for r in scored),
                'subjects':scored,'native_failures':prediction['native_failures'],
                'dose_log_rmse':sum(r['log_rmse'] for r in scored)/len(subjects) if complete else None,
                **summarize_subject_metrics(scored,len(subjects))}
    for selection in config['selections']:
        rows=[]
        for case in checked:
            scored=unique_cases.get((case['case_id'],selection['model_key']))
            row={**selection,'case_id':case['case_id'],'cohort_id':case['design']['cohort_id'],
                'publication_cluster_id':case['design']['publication_cluster_id'],'compound':case['design']['compound'],
                'reference_subjects_expected':case['design']['population']['n'],
                'status':scored['status'] if scored else 'failed_retained_in_denominator',
                'dose_log_rmse':scored['dose_log_rmse'] if scored else None,
                **{name:scored[name] if scored else None for name in
                    ('gmfe','fold_bias','within_2fold_fraction','mean_log_ratio','mean_absolute_log_ratio')},
                'strict_endpoint_summary':scored['strict_endpoint_summary'] if scored else
                    summarize_subject_metrics([],case['design']['population']['n'])['strict_endpoint_summary'],
                'censored_pk_summary':scored['censored_pk_summary'] if scored else
                    summarize_subject_metrics([],case['design']['population']['n'])['censored_pk_summary'],
                'failure':selection.get('failure') if scored is None else
                    ('native_or_subject_prediction_failed' if scored['status']!='evaluated' else None)}
            rows.append(row);method_cases.append(row)
        complete=all(r['status']=='evaluated' for r in rows)
        signed=sum(r['mean_log_ratio'] for r in rows)/2 if complete else None
        absolute=sum(r['mean_absolute_log_ratio'] for r in rows)/2 if complete else None
        endpoints=[r['strict_endpoint_summary'] for r in rows]
        endpoint_complete=all(e['subjects_evaluable']>0 for e in endpoints)
        endpoint_summary={'subjects_expected':sum(e['subjects_expected'] for e in endpoints),
            'subjects_evaluable':sum(e['subjects_evaluable'] for e in endpoints),
            'coverage':sum(e['subjects_evaluable'] for e in endpoints)/sum(e['subjects_expected'] for e in endpoints),
            'conditional_on_complete_original_grid':True,'both_dose_cohorts_represented':endpoint_complete,
            **{name+'_mean_absolute_log_error':sum(e[name+'_mean_absolute_log_error'] for e in endpoints)/2
                if endpoint_complete else None for name in ('sampled_cmax','observed_window_auc')}}
        censored=[r['censored_pk_summary'] for r in rows]
        censored_summary={'subjects_expected':sum(e['subjects_expected'] for e in censored),
            'sampled_cmax_evaluable':sum(e['sampled_cmax_evaluable'] for e in censored),
            'sampled_cmax_coverage':sum(e['sampled_cmax_evaluable'] for e in censored)/sum(e['subjects_expected'] for e in censored),
            'auc_interval_evaluable':sum(e['auc_interval_evaluable'] for e in censored),
            'auc_interval_coverage':sum(e['auc_interval_evaluable'] for e in censored)/sum(e['subjects_expected'] for e in censored),
            **{name:sum(e[name] for e in censored)/2 if all(e[name] is not None for e in censored) else None
                for name in ('sampled_cmax_mean_absolute_log_error','auc_mean_distance_over_upper_bound','auc_mean_interval_width')},
            'conditional_on_endpoint_eligibility':True,'both_cohorts_required_for_error_summary':True,
            'interval_compatibility_is_not_point_accuracy':True}
        method_studies.append({**selection,'compound':rows[0]['compound'],'publication_cluster_id':rows[0]['publication_cluster_id'],
            'dose_cohorts_expected':2,'dose_cohorts_evaluated':sum(r['status']=='evaluated' for r in rows),
            'status':'evaluated' if complete else 'failed_retained_in_denominator',
            'study_log_rmse':sum(r['dose_log_rmse'] for r in rows)/2 if complete else None,
            'gmfe':math.exp(absolute) if complete else None,'fold_bias':math.exp(signed) if complete else None,
            'mean_log_ratio':signed,'mean_absolute_log_ratio':absolute,
            'within_2fold_fraction':sum(r['within_2fold_fraction'] for r in rows)/2 if complete else None,
            'strict_endpoint_summary':endpoint_summary,
            'censored_pk_summary':censored_summary,
            'drug_bootstrap_confidence_interval':None,'CI_reason':'One independent drug/publication; cohorts, subjects and seeds do not increase the drug count'})
    strata=[]
    for block,(budget,dimension,methods) in BLOCKS.items():
        for method in methods:
            rows=[r for r in method_studies if r['block_id']==block and r['method']==method]
            complete=all(r['status']=='evaluated' for r in rows)
            strata.append({'block_id':block,'method':method,'solver_budget':budget,'exact_free_parameter_count':dimension,
                'seeds_expected':[r['seed'] for r in rows],'seeds_evaluated':sum(r['status']=='evaluated' for r in rows),
                'mean_across_registered_seeds':sum(r['study_log_rmse'] for r in rows)/len(rows) if complete else None,
                'gmfe':math.exp(sum(r['mean_absolute_log_ratio'] for r in rows)/len(rows)) if complete else None,
                'fold_bias':math.exp(sum(r['mean_log_ratio'] for r in rows)/len(rows)) if complete else None,
                'within_2fold_fraction':sum(r['within_2fold_fraction'] for r in rows)/len(rows) if complete else None,
                'strict_endpoint_summaries_by_seed':{str(r['seed']):r['strict_endpoint_summary'] for r in rows},
                'censored_pk_summaries_by_seed':{str(r['seed']):r['censored_pk_summary'] for r in rows},
                'success_only_mean':False,'selection_ids':[r['selection_id'] for r in rows]})
    output.mkdir(parents=True);sealed_dir=output/'sealed';sealed_dir.mkdir(mode=0o700)
    scores={'schema_version':'pbpk-all-method-independent-curve-scores-v1','unique_model_cases':list(unique_cases.values()),
        'method_cases':method_cases,'method_studies':method_studies,'method_budget_strata':strata,
        'independent_drugs':1,'independent_studies':1,'method_jobs_expected':57,'method_dose_rows_expected':114,
        'aggregation':'subject natural-log RMSE mean -> equal dose mean -> all registered seed mean within the same method/budget/dimension',
        'prediction_is_group_reference_not_individualized':True,'prediction_clipping':False,'observation_imputation':False,
        'scores_sealed_from_method_development':True,'prediction_lock_sha256':sha(lock_path),
        'evaluation_contract':EVALUATION_CONTRACT}
    score_path=sealed_dir/'all_method_curve_scores.json';score_path.write_text(json.dumps(scores,indent=2,allow_nan=False)+'\n');score_path.chmod(0o600)
    receipt={'status':'scored_and_sealed','method_jobs_expected':57,'method_dose_rows_expected':114,
        'method_dose_rows_evaluated':sum(r['status']=='evaluated' for r in method_cases),
        'method_dose_rows_failed_retained':sum(r['status']!='evaluated' for r in method_cases),
        'unique_models':len(config['models']),'reference_subjects_expected':sum(c['design']['population']['n'] for c in checked),
        'independent_drugs':1,'independent_studies':1,'method_budget_strata':len(strata),
        'sealed_scores_path':str(score_path.relative_to(ROOT)),'sealed_scores_sha256':sha(score_path),
        'lock_path':str(lock_path.relative_to(ROOT)),'lock_sha256':sha(lock_path),
        'metrics_exposed_to_method_development':False,'scorer_sha256':sha(Path(__file__))}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--lock',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--validate-only',action='store_true',help='Validate all predictions without opening sealed observations')
    args=parser.parse_args()
    if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    if args.validate_only:
        _,config,cases=validate_locked_inputs(local(args.lock))
        print(json.dumps({'status':'lock_validated','method_jobs':len(config['selections']),'dose_cases':len(cases),'outcomes_read':False}));return
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    print(json.dumps(score_locked(local(args.lock),local(args.output)),indent=2))


if __name__=='__main__':main()
