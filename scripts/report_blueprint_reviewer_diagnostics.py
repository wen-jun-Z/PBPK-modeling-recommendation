#!/usr/bin/env python3
"""Aggregate the predeclared diagnostic reruns without altering frozen results."""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
import blueprint_reviewer_diagnostics as d

DENSE_FIELDS={'auc':('value_mg_h_per_l','mg.h/L'),'cmax':('value_mg_per_l','mg/L'),
    'tmax':('value_h','h'),'c_last':('value_mg_per_l','mg/L'),'terminal_half_life':('value_h','h'),
    'auc_inf':('value_mg_h_per_l','mg.h/L'),'clearance':('value_l_per_h','L/h'),
    'vz':('value_l','L'),'mrt':('value_h','h'),'vss':('value_l','L')}


def dense_ranges(members):
    output={}
    for ep,(field,unit) in DENSE_FIELDS.items():
        records=[m['dense_endpoints'][ep] for m in members]
        vals=[e[field] for e in records if e.get('status')=='completed' and e.get(field) is not None]
        reasons={}
        for e in records:
            if e.get('status')!='completed':reasons[e.get('reason','unavailable')]=reasons.get(e.get('reason','unavailable'),0)+1
        output[ep]=dict(unit=unit,members_total=len(members),estimable_members=len(vals),
            minimum=min(vals) if vals else None,maximum=max(vals) if vals else None,
            max_over_min=max(vals)/min(vals) if vals and min(vals)>0 else None,unavailable_reasons=reasons,
            reference_status='model-only; not a clinical reference or calibrated prediction interval')
    return output


def ranges(members, field):
    values={}
    for member in members:
        for key,value in member.get(field,{}).items():
            if isinstance(value,(int,float)) and math.isfinite(value) and value>0:values.setdefault(key,[]).append(value)
    return {k:dict(n=len(v),minimum=min(v),maximum=max(v),max_over_min=max(v)/min(v)) for k,v in values.items()}


def report():
    p=d.read(d.RUN/'protocol.json');s=d.read(d.RUN/'state.json');quality=d.read(d.RUN/'nca_quality.json')
    if len(s['profile_done'])!=len(p['profile_tasks']):raise ValueError('All five-point profiles must finish before reporting')
    _,source,_,_=d.source_bundle();comparisons=[]
    for key,source_ref in s['profile_done'].items():
        five=d.checked(source_ref);cid=five['job']['case_id'];old_ref=source['designs'][cid+'__late_four']['source']
        four=d.checked(old_ref);m4=four['members'];m5=five['members']
        p4=np.exp(np.asarray([m['log_parameters'] for m in m4]));p5=np.asarray([m['parameters'] for m in m5])
        log_width4=np.log(p4.max(axis=0)/p4.min(axis=0));log_width5=np.log(p5.max(axis=0)/p5.min(axis=0))
        four_values={k:dict(n=len([x for x in v if x is not None]),minimum=min(x for x in v if x is not None),
            maximum=max(x for x in v if x is not None)) for k,v in four['goal_values'].items() if any(x is not None for x in v)}
        four_profiles={k:d.checked(v) for k,v in four['profile_sources'].items()}
        five_profiles={k:d.checked(v) for k,v in five['profiles'].items()}
        status=lambda profiles:{k:dict(status=v['status'],nodes=len(v['profile']),converged_nodes=sum(r['successful'] for r in v['profile']),
            calls=v['calls'],failures=v['failures']) for k,v in profiles.items()}
        with np.load(ROOT/five['curves']['path']) as arrays:
            tt=arrays['time_h'];lower=arrays['minimum_mg_l'];upper=arrays['maximum_mg_l']
            supported=(tt>0)&(lower>0);width=upper[supported]/lower[supported]
            envelope=dict(artifact=five['curves'],units='mg/L',time_unit='h',points=len(tt),
                horizon_h=float(tt[-1]),members=len(m5),minimum_field='minimum_mg_l',maximum_field='maximum_mg_l',
                all_member_curves_field='concentrations_mg_l',largest_upper_lower_ratio=float(width.max()) if len(width) else None,
                interval_type='pointwise min/max of finite data-compatible profiles; not a confidence or prediction interval')
        comparisons.append(dict(case_id=cid,methods=five['job']['methods'],five_source=source_ref,four_source=old_ref,
            members_four=len(m4),members_five=len(m5),threshold_four=four['threshold'],threshold_five=five['threshold'],
            published_five_point_best_data_mse=five['original_best_parameters']['development_ln_rmse']**2,
            published_best_within_profile_tolerance=five['original_best_parameters']['development_ln_rmse']**2<=five['threshold'],
            profile_bounds_changed_between_four_and_five=any(
                not np.allclose(four_profiles[k]['lower4'],five_profiles[k]['lower4'],rtol=0,atol=1e-12)
                or not np.allclose(four_profiles[k]['upper4'],five_profiles[k]['upper4'],rtol=0,atol=1e-12)
                for k in four_profiles),
            parameter_min_four=p4.min(axis=0).tolist(),parameter_max_four=p4.max(axis=0).tolist(),
            parameter_min_five=p5.min(axis=0).tolist(),parameter_max_five=p5.max(axis=0).tolist(),
            log_parameter_width_four=log_width4.tolist(),log_parameter_width_five=log_width5.tolist(),
            parameter_width_reduced=(log_width5<log_width4-1e-10).tolist(),
            four_target_goal_ranges=four_values,five_matched_endpoint_ranges=ranges(m5,'matched_endpoints'),
            five_dense_endpoint_ranges=dense_ranges(m5),target_curve_envelope=envelope,
            four_profile_status=status(four_profiles),five_profile_status=status(five_profiles),
            note='Different visible-data fits, same fixed MSE tolerance; coordinate-profile support is neither a joint confidence region nor formal proof of identifiability. If repair changes bounds, width changes cannot be attributed solely to adding a point; the penalized published optimum may fall outside this data-only set.'))
    cohorts={};boundaries={}
    for method in ('time_repaired','stable_repaired'):
        rr=[]
        for case in p['cases']:
            available=[r for r in quality['cases'] if r['case_id']==case['case_id']]
            r=next((r for r in available if r['method']==method),available[0]);rr.append(r)
        vals=[];eligible=[]
        for r in rr:
            ep=r['original_endpoints']['terminal_half_life']
            if ep['error'] is not None:vals.append(ep['error'])
            if r['strict_matched']['terminal_half_life_qualified'] and r['strict_observed']['terminal_half_life_qualified']:
                eligible.append(dict(case_id=r['case_id'],drug=r['drug'],error=ep['error'],prediction=ep['value'],reference=ep['reference']))
        gm=lambda v:math.exp(sum(math.log(x) for x in v)/len(v)) if v else None
        cohorts[method]=dict(drugs=len(rr),original_half_life_gmfe=gm(vals),strict_half_life_drugs=len(eligible),
            strict_half_life_gmfe=gm([r['error'] for r in eligible]),strict_records=eligible,
            poor_matched_r2=[r['drug'] for r in rr if r['strict_matched']['adjusted_r_squared'] is None or r['strict_matched']['adjusted_r_squared']<.98],
            poor_reference_r2=[r['drug'] for r in rr if r['strict_observed']['adjusted_r_squared'] is None or r['strict_observed']['adjusted_r_squared']<.98],
            dense_numeric_qualified=sum(r['strict_dense']['terminal_half_life_qualified'] for r in rr),
            numeric_qualification_is_not_clinical_validation=True)
        selected=[]
        for case in p['cases']:
            available=[r for r in quality['parameters'] if r['case_id']==case['case_id']]
            record=next((r for r in available if r['method']==method),available[0])
            fit=next(f for f in record['fits'] if f['selected'])
            distances=fit['normalized_boundary_distance']
            selected.append(dict(case_id=case['case_id'],drug=case['drug'],formula=fit['formula'],
                normalized_log_distance=distances,near_boundary=[v<=.01 for v in distances],
                at_boundary=[v<=1e-6 for v in distances]))
        boundaries[method]=dict(n=len(selected),near_boundary_counts={a:sum(r['near_boundary'][i] for r in selected) for i,a in enumerate(d.AXES)},
            at_boundary_counts={a:sum(r['at_boundary'][i] for r in selected) for i,a in enumerate(d.AXES)},
            any_near_boundary=sum(any(r['near_boundary']) for r in selected),any_at_boundary=sum(any(r['at_boundary']) for r in selected),
            definition='d=min((ln(theta)-ln(lower))/(ln(upper)-ln(lower)),1-that position); near d<=0.01; at d<=1e-6; numerical search bounds, not biological acceptability',
            rows=selected)
    labels=quality['original_half_life_training_label_audit'];by_design={}
    for r in labels:by_design.setdefault(r['design_key'],[]).append(r)
    label_summary=dict(records=len(labels),strictly_qualified_records=sum(r['label_eligible'] for r in labels),
        designs=len(by_design),fully_qualified_designs=[k for k,v in by_design.items() if all(r['label_eligible'] for r in v)],
        affected_designs=[k for k,v in by_design.items() if not all(r['label_eligible'] for r in v)],
        instruction='No relabeling or retraining performed here. Half-life learned selectors need a separately versioned data-quality revision; Cmax labels do not use this tail fit.')
    windows=[]
    for record in quality['cases']:
        unit=d.checked(record['source']);request=d.checked(unit['request']);native=d.checked(unit['prediction'])
        mw=next(e['value'] for e in request['evidence'] if e['parameter']=='molecular_weight')
        tt,yy,_=d.prediction_in_hours(native,mw);fit=record['dense'].get('terminal_fit') or {}
        lower=fit.get('requested_fit_start_h');upper=fit.get('requested_fit_end_h')
        actual=[(t,y) for t,y in zip(tt,yy) if lower is not None and lower<=t<=upper]
        windows.append(dict(case_id=record['case_id'],drug=record['drug'],method=record['method'],
            model_dense_requested_window_h=[lower,upper],model_dense_actual_sample_window_h=[actual[0][0],actual[-1][0]] if actual else None,
            model_dense_actual_concentration_ng_ml=[actual[0][1],actual[-1][1]] if actual else None,
            matched_tail=record['matched'].get('selected_tail'),observed_tail=record['observed'].get('selected_tail'),
            lloq_ng_ml=record['clinical_contract']['cutoff_ng_ml'],cutoff_basis=record['clinical_contract']['cutoff_basis']))
    out=dict(version=d.VERSION,source_protocol=d.ref(d.RUN/'protocol.json'),quality_source=d.ref(d.RUN/'nca_quality.json'),
        methods=cohorts,parameter_order=d.AXES,parameter_boundaries=boundaries,profile_comparisons=comparisons,half_life_training_label_audit=label_summary,
        terminal_regression_windows=windows,
        total_profile_calls=sum(d.checked(r)['calls'] for r in s['profile_done'].values()),
        interpretation='Finite data-compatible sets expose parameter compensation and endpoint uncertainty, not calibrated prediction bands or biological parameter recovery')
    rf_path=d.RUN/'rf_ablation/results.json'
    if rf_path.exists():
        from report_blueprint_validation import clustered_geomean
        rf=d.read(rf_path);clusters={c['case_id']:c['cluster_id'] for c in p['cases']};rf_pairs={}
        for mode in ('median_cl','median_fu','median_both'):
            rf_pairs[mode]={}
            for scope in ('all_drugs','input_actually_changed'):
                rr=[r for r in rf['rows'] if r['mode']==mode and (scope=='all_drugs' or r['changed_parameters']!='original')]
                rf_pairs[mode][scope]={}
                for ep in ('curve','cmax','auc','terminal_half_life'):
                    values={}
                    for r in rr:
                        if r['score'] is None:continue
                        a=r['score']['curve_metrics']['gmfe'] if ep=='curve' else r['score']['endpoints'][ep]['error']
                        b=r['baseline_score']['curve_metrics']['gmfe'] if ep=='curve' else r['baseline_score']['endpoints'][ep]['error']
                        if a is not None and b is not None:values[r['case_id']]=a/b
                    rf_pairs[mode][scope][ep]=clustered_geomean(values,{k:clusters[k] for k in values},seed=20261009,resamples=2000)
        out['rf_ablation']=dict(source=d.ref(rf_path),summary=rf['summary'],paired_ratios=rf_pairs,
            ratio_direction='median input / RF input error; >1 favors RF, <1 favors median; existing independent inputs remain unchanged')
    d.save(d.RUN/'report.json',out)
    print(json.dumps(dict(status='complete',profile_comparisons=len(comparisons),methods=cohorts,
        label_audit=label_summary,total_profile_calls=out['total_profile_calls']),ensure_ascii=False))


if __name__=='__main__':report()
