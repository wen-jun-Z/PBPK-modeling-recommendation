#!/usr/bin/env python3
"""Diagnose frozen sampling choices without refitting or selecting a new model.

Outputs candidate-vs-time features/endpoint gains and descriptive diagnostic
counts. Oracle values use target answers only to measure opportunity; they are
never deployable predictions. Montelukast is kept but marked as an incomplete
repair menu, never mixed with repaired entries in an oracle.
"""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/blueprint_design/sampling_stability_20261009'
OUT=ROOT/'experiments/blueprint_design/sampling_policy_20261009/diagnosis'
ENDPOINTS=('cmax','curve','auc','c_last','terminal_half_life')
FEATURES=('log_time_over_infusion','nearest_visible_log_time_distance',
 'mean_log_concentration_over_visible_max','concentration_log_variance',
 'linear_explained_goal_variance','nonlinear_explained_goal_variance',
 'local_c_optimal_fractional_reduction','after_infusion')
TOL=1e-10


def ref(path):return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def read(path):return json.loads(path.read_text())
def gm(v):return math.exp(math.fsum(math.log(x) for x in v)/len(v)) if v else None

def error(record,endpoint):
    value=record['curve_metrics']['gmfe'] if endpoint=='curve' else record['endpoints'][endpoint]['error']
    return float(value) if value is not None and math.isfinite(value) and value>0 else None

def loggain(a,b):return math.log(a/b) if a is not None and b is not None else None

def counts(values):
    vv=[v for v in values if v is not None]
    return dict(n=len(vv),better=sum(v>TOL for v in vv),worse=sum(v<-TOL for v in vv),tied=sum(abs(v)<=TOL for v in vv))

def correlation(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if len(a)<3 or np.ptp(a)<1e-12 or np.ptp(b)<1e-12:return None
    return float(spearmanr(a,b).statistic)


def main():
    from report_sampling_stability import load_complete
    state,p,completed,endpointrows=load_complete()
    features=read(RUN/'features.json')['designs'];choices=read(RUN/'choices.json')['choices']
    path=ROOT/'experiments/blueprint_design/sampling_expansion_20261009/results.json'
    records=read(path)['records'];record={(r['design_key'],r['candidate']):r for r in records}
    cases={c['case_id']:c for c in p['cases']};original=set(p['original_case_ids'])
    rows=[];menus=[]
    for key,f in features.items():
        cid=key.rsplit('__',1)[0];case=cases[cid];regime=key.rsplit('__',1)[1]
        quality='incomplete_repair_menu' if case['drug']=='Montelukast' else 'uniform_predictor'
        for goal in ('cmax','terminal_half_life'):
            decision=choices[key][goal]['decision'];detail=f['goals'][goal]
            if detail['status']!='selected':raise ValueError('Unsupported required feature menu')
            x=np.asarray(detail['X']);ti,pi,si=(decision[k] for k in ('time_index','proposed_index','selected_index'))
            candidates=[record[key,key+f'__point_{i:02d}'] for i in f['candidate_ids']]
            baseline=record[key,'baseline'];tr=candidates[ti];pr=candidates[pi];sr=candidates[si]
            if len(candidates)!=len(x):raise ValueError('Unaligned candidate and feature rows')
            scope='original_ten' if cid in original else 'added_twenty'
            actual=np.array([error(r,goal) for r in candidates],float)
            if not np.isfinite(actual).all():raise ValueError('Missing requested error should not be silently filled')
            oi=int(min(range(len(actual)),key=lambda i:(actual[i],f['candidate_times_h'][i])))
            oracle=candidates[oi];nonanchors=[i for i in range(len(x)) if i!=ti]
            menu=dict(case_id=cid,drug=case['drug'],cluster_id=case['cluster_id'],cohort=scope,
                design_key=key,regime=regime,goal=goal,quality=quality,candidate_count=len(x),
                time_index=ti,proposal_index=pi,stable_index=si,oracle_index=oi,reason=decision['reason'],
                time_h=f['candidate_times_h'][ti],proposal_time_h=f['candidate_times_h'][pi],stable_time_h=f['candidate_times_h'][si],
                proposal_gain={e:loggain(error(tr,e),error(pr,e)) for e in ENDPOINTS},
                stable_gain={e:loggain(error(tr,e),error(sr,e)) for e in ENDPOINTS},
                oracle_gain={e:loggain(error(tr,e),error(oracle,e)) for e in ENDPOINTS},
                baseline_error={e:error(baseline,e) for e in ENDPOINTS},time_error={e:error(tr,e) for e in ENDPOINTS},
                oracle_requested_error=error(oracle,goal),
                best_curve_error=min(error(r,'curve') for r in candidates),
                per_menu_rank_spearman=correlation(decision.get('full_scores',[]),[-math.log(a) for a in actual]) if decision.get('full_scores') else None,
                allowed_major_curve_degradation_fraction=.05,
                requested_gain_available_without_curve_regression=any(error(r,goal)<error(tr,goal)*(1-TOL) and error(r,'curve')<=error(tr,'curve')*(1+TOL) for r in candidates),
                model_condition='Old uniform predictor for oracle; current repair may alter Montelukast choices',
                margins=decision['margins'])
            menus.append(menu)
            for ix,candidate in enumerate(candidates):
                row=dict(case_id=cid,drug=case['drug'],cohort=scope,cluster_id=case['cluster_id'],design_key=key,regime=regime,
                    goal=goal,quality=quality,candidate_id=f['candidate_ids'][ix],candidate_index=ix,time_h=f['candidate_times_h'][ix],
                    is_time=ix==ti,is_proposal=ix==pi,is_stable=ix==si,is_oracle=ix==oi,
                    policy_reason=decision['reason'],old_predicted_margin=(decision['full_scores'][ix]-decision['full_scores'][ti]) if decision.get('full_scores') else None)
                for j,name in enumerate(FEATURES):row['dx_'+name]=float(x[ix,j]-x[ti,j])
                for e in ENDPOINTS:
                    row['gain_'+e]=loggain(error(tr,e),error(candidate,e))
                    row['error_'+e]=error(candidate,e)
                ep=candidate['endpoints']['terminal_half_life'];te=tr['endpoints']['terminal_half_life']
                row['half_life_qualified']=(not ep['weak_tail'] and not ep['reference_weak_tail'] and not te['weak_tail'] and not te['reference_weak_tail'])
                rows.append(row)
    scopes={};features_summary={}
    for regime in ('early_four','late_four'):
        for goal in ('cmax','terminal_half_life'):
            for scope in ('original_ten','added_twenty','all_thirty'):
                full=[m for m in menus if m['regime']==regime and m['goal']==goal and (scope=='all_thirty' or m['cohort']==scope)]
                eligible=[m for m in full if m['quality']=='uniform_predictor']
                groups=Counter(m['reason'] for m in full)
                blocked=[m for m in eligible if m['reason']=='nonpositive_or_tied_study_margin']
                accepted=[m for m in eligible if m['reason']=='unanimous_positive_margin']
                propdiff=[m for m in eligible if m['proposal_index']!=m['time_index']]
                cand=[r for r in rows if r['regime']==regime and r['goal']==goal and (scope=='all_thirty' or r['cohort']==scope) and r['quality']=='uniform_predictor' and not r['is_time']]
                conf=[r for r in cand if r['gain_'+goal]>TOL and r['gain_curve']<-TOL]
                conf5=[r for r in cand if r['gain_'+goal]>TOL and r['gain_curve']<-math.log(1.05)]
                rp=[m['per_menu_rank_spearman'] for m in eligible if m['per_menu_rank_spearman'] is not None]
                vals=dict(all_menus=len(full),uniform_predictor_menus=len(eligible),decision_reasons=dict(groups),
                    proposed_changes=counts([m['proposal_gain'][goal] for m in propdiff]),
                    guard_rejected_changes=counts([m['proposal_gain'][goal] for m in blocked]),
                    guard_accepted_changes=counts([m['proposal_gain'][goal] for m in accepted]),
                    rejected_helpful_proposals=[m['drug'] for m in blocked if m['proposal_gain'][goal]>TOL],
                    rejected_harmful_proposals=[m['drug'] for m in blocked if m['proposal_gain'][goal]<-TOL],
                    accepted_harmful_proposals=[m['drug'] for m in accepted if m['proposal_gain'][goal]<-TOL],
                    proposed_requested_improvement_but_curve_worse=sum(m['proposal_gain'][goal]>TOL and m['proposal_gain']['curve']<-TOL for m in propdiff),
                    candidate_pairs=len(cand),requested_improving_pairs=sum(r['gain_'+goal]>TOL for r in cand),
                    requested_improving_but_curve_worse_pairs=len(conf),requested_improving_but_curve_worse_over5pct=len(conf5),
                    time_requested_gmfe=gm([m['time_error'][goal] for m in eligible]),
                    oracle_requested_gmfe=gm([m['oracle_requested_error'] for m in eligible]),
                    oracle_requested_ratio=math.exp(np.mean([-m['oracle_gain'][goal] for m in eligible])),
                    oracle_gain_over10pct_drugs=sum(m['oracle_gain'][goal]>math.log(1/.9) for m in eligible),
                    oracle_helps_but_curve_worse=sum(m['oracle_gain'][goal]>TOL and m['oracle_gain']['curve']<-TOL for m in eligible),
                    oracle_curve_gmfe=gm([m['best_curve_error'] for m in eligible]),
                    time_curve_gmfe=gm([m['time_error']['curve'] for m in eligible]),
                    requested_gain_without_curve_regression_drugs=sum(m['requested_gain_available_without_curve_regression'] for m in eligible),
                    median_menu_rank_spearman=float(np.median(rp)) if rp else None)
                scopes['__'.join((regime,goal,scope))]=vals
    for regime in ('early_four','late_four'):
        for goal in ('cmax','terminal_half_life'):
            tr=[r for r in rows if r['regime']==regime and r['goal']==goal and r['cohort']=='original_ten' and not r['is_time']]
            te=[r for r in rows if r['regime']==regime and r['goal']==goal and r['cohort']=='added_twenty' and not r['is_time']]
            s=[]
            for feature in FEATURES:
                name='dx_'+feature;a=np.array([v[name] for v in tr]);b=np.array([v[name] for v in te]);lo,hi=float(min(a)),float(max(a))
                s.append(dict(feature=feature,original_range=[lo,hi],added_range=[float(min(b)),float(max(b))],
                    fraction_added_outside_original_range=float(np.mean((b<lo)|(b>hi))),
                    original_spearman_with_requested_gain=correlation(a,[v['gain_'+goal] for v in tr]),
                    added_spearman_with_requested_gain=correlation(b,[v['gain_'+goal] for v in te])))
            features_summary[regime+'__'+goal]=s
    # Exact algebraic audit: label/feature shifts followed by the same menu
    # centering CANNOT change the ranker; avoid claiming this as an innovation.
    center_error=0.
    for key,f in features.items():
        goal='cmax';x=np.array(f['goals'][goal]['X']);d=choices[key][goal]['decision'];ti=d['time_index'];y=np.array([math.log(record[key,'baseline']['endpoints'][goal]['error']/record[key,key+f'__point_{i:02d}']['endpoints'][goal]['error']) for i in f['candidate_ids']])
        a=x-x.mean(axis=0);z=x-x[ti];b=z-z.mean(axis=0);yy=y-y[ti]
        center_error=max(center_error,float(np.max(abs(a-b))),float(np.max(abs((y-y.mean())-(yy-yy.mean())))))
    result=dict(version='frozen-sampling-policy-diagnosis-1',created_at=time.time(),
        sources={name:ref(path) for name,path in [('current_protocol',RUN/'protocol.json'),('current_features',RUN/'features.json'),('current_choices',RUN/'choices.json'),('exhaustive_outcomes',path),('diagnostic_script',Path(__file__).resolve())]},
        no_new_model_fit=True,feature_names=list(FEATURES),menus=menus,summary=scopes,feature_transfer=features_summary,
        row_count=len(rows),nonidentical_feature_label_shift_after_centering_max_abs_error=center_error,
        limits=['Oracle uses clinical target answers and is hindsight only. It does not show online selectability.',
         'Montelukast retained and flagged; complete current-repair candidate menu is unavailable, excluded only from comparable oracle/count diagnostics, not primary evaluation.',
         'Uniform candidates use completed pre-repair predictor; for other29 repair did not activate.',
         'Point-level rows are not independent drugs: 30drugs,29target-study clusters,60layouts. Original ranker trained10drugs/9clusters.',
         'New labels remain retrospective development labels. Grouped30 crossvalidation is development validation, not a new untouched test.',
         'Half-life labels with weak observed/predicted tails are separately flagged and unsuitable as strong physiology targets.',
         'Spearman feature associations are descriptive; do not select features by added20 target correlation.'],
        method_recommendation=dict(name='Visible-context conditional action-value policy',
         core='Predict requested relative error gain vs time action; add a separate curve-degradation head/constraint. Do not merely relabel a within-menu centered regression.',
         representation='Candidate-relative features interacted with a few frozen visible-data contexts; contexts derived only from initial four concentrations/model residuals and dose/infusion.',
         policy='Requested benefit objective subject to predeclared curve deterioration tolerance, with time fallback when unsupported. No arbitrary weighted composite reported.',
         original10_track='Fit only original10 historical labels with leave-target-study-out inner selection. Evaluate added20 as existing retrospective extension, not untouched data.',
         grouped30_track='All layouts/candidates/goals of the held-out drug and shared study stay together. Any model/threshold tuning must be inner grouped; report outer predictions once.',
         discipline='One prespecified simple method plus time/current ranker controls; no broad model sweep followed by best-result selection. Freeze before independent next cohort.'))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'candidate_pairs.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for key in ('late_four__cmax__original_ten','late_four__cmax__added_twenty','early_four__cmax__added_twenty'):
        print(key,json.dumps(scopes[key],ensure_ascii=False))
    print('diagnosis',OUT,'rows',len(rows),'centering_shift_error',center_error)

if __name__=='__main__':main()
