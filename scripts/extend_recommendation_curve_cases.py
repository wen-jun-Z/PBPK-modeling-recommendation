#!/usr/bin/env python3
"""Score independently digitized observations against previously locked predictions.

This is a retrospective illustrative extension, never a replacement for frozen
cohort statistics. No model execution, fitting, training or reranking occurs.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

from model_recommendation_paper import METHODS
from pbpk_match_light_curves import positions, curve_errors, score_position
from render_recommendation_curve_cases import load_figure_data, OWN_METHODS, COMPONENT_METHOD

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'experiments/model_recommendation/curve_case_extension_20261027'
RUN = ROOT / 'experiments/model_recommendation/execution_validation_20261001'
OWN = ROOT / 'experiments/model_recommendation/pbpk_match_light'
COMP = ROOT / 'experiments/model_recommendation/component_integration_20261007'
DEST = ROOT / 'results/data/model_recommendation_curve_cases_extended.json'
SID = 'clinical:metformin_pmc12394267_reference_fed'
IDS = ['b9077ad9-89c1-5ab3-8897-c9c6721c73a1',
       '720e982a-5b38-582c-82d2-8859265f5c5c',
       'e2cf231e-6b77-5219-a90f-53e4a35836e6']


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return {'path': str(Path(path).relative_to(ROOT)),
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def verify_refs(refs):
    for r in refs:
        assert ref(ROOT / r['path'])['sha256'] == r['sha256'], r['path']


def bounded_gmfe(pred, low, high):
    """Exact independent observation-interval extrema, not a confidence interval."""
    p, lo, hi = np.log(pred), np.log(low), np.log(high)
    dmin = np.maximum(np.maximum(lo-p, p-hi), 0)
    dmax = np.maximum(np.abs(p-lo), np.abs(p-hi))
    return [float(np.exp(dmin.mean())), float(np.exp(dmax.mean()))]


def main():
    protocol = read(EXT/'protocol.json')
    assert ref(EXT/'protocol.json')['sha256'] == 'cc5c5a9867d1f6bcdf0cde4f20f6c8f21556bfa7f0d7a39d6f1acc218cfca17e'
    verify_refs(protocol['pinned_inputs'])
    amendment = read(EXT/'protocol_amendment_01.json')
    verify_refs(amendment['pinned_inputs'])
    observation_path = EXT/'observations_metformin.json'
    obs = read(observation_path)
    ot = np.asarray(obs['time_h'], float)
    oy = np.asarray(obs['concentration_ng_ml'], float)
    lo = np.asarray(obs['concentration_lower_ng_ml'], float)
    hi = np.asarray(obs['concentration_upper_ng_ml'], float)
    listed = np.asarray(obs['methods_listed_times_mask'], bool)
    assert len(ot) == len(oy) == len(lo) == len(hi)
    assert np.all(np.diff(ot)>0) and np.all(lo>0) and np.all(lo<=oy) and np.all(oy<=hi)
    # Seal the source extraction before any model-error calculation.
    extraction_receipt = EXT/'observations_lock.json'
    obs_ref = ref(observation_path)
    if extraction_receipt.exists():
        assert read(extraction_receipt)['observation'] == obs_ref
    else:
        write(extraction_receipt, {'observation':obs_ref, 'protocol':ref(EXT/'protocol.json'),
            'extraction_independent_of_candidate_predictions':True,
            'all_digitized_points_retained':True, 'sealed_before_first_metric_calculation':True})
    candidates = {c['candidate_id']: c for c in
                  map(json.loads,(RUN/'rankings/inputs/candidates.jsonl').read_text().splitlines())}
    own_protocol = read(OWN/'protocol.json')
    qi = own_protocol['query_ids'].index(SID)
    folds = [i for i,f in enumerate(own_protocol['folds']) if qi in f['outer_test']]
    assert len(folds)==1
    fold=own_protocol['folds'][folds[0]]
    assert all(own_protocol['groups'][i] != own_protocol['groups'][qi] for i in fold['outer_train'])
    global_lock=read(RUN/'native/predictions_lock.json')
    jobs={j['candidate_id']:j for j in global_lock['jobs'] if j['scenario_id']==SID}
    assert set(jobs)==set(IDS)
    case={'compound':'metformin','scenario_id':SID,'dose_mg':250,
          'active_moiety_dose_mg':195,'route':'oral','food':'fed_30_min_after_meal',
          'formulation':'Glycoran reference tablet','matrix':'plasma','analyte':'parent',
          'binding':'total','publication_cluster_id':'doi:10.1007/s13318-025-00961-4',
          'cohort_id':'trial:CTR20180793:fed:reference','observation':obs,
          'curves':[], 'rankings':[], 'evaluation_stratum':'post_hoc_reference_diagnostic'}
    for i,cid in enumerate(IDS,1):
        job=jobs[cid]
        pred_ref=next(r for r in job['artifacts'] if r['path'].endswith('/prediction.json'))
        verify_refs([pred_ref])
        pred=read(ROOT/pred_ref['path'])
        assert pred['scenario_id']==SID and pred['candidate_id']==cid
        t=np.asarray(pred['time_h'],float); y=np.asarray(pred['concentration_ng_ml'],float)
        assert np.all(np.diff(t)>0) and np.isfinite(y).all() and np.all(y>=0)
        assert t[0]<=ot[0] and t[-1]>=ot[-1]
        yp=np.interp(ot,t,y)
        assert np.all(yp>0)
        errors=np.log(yp/oy)
        c=candidates[cid]
        curve={'label':f'M{i}','candidate_id':cid,'name':c['family_name'],
               'family_id':c['family_id'],'version':c.get('version',''),
               'time_h':t.tolist(),'concentration_ng_ml':y.tolist(),
               'matched_time_h':ot.tolist(),'matched_concentration_ng_ml':yp.tolist(),
               'gmfe':float(np.exp(np.abs(errors).mean())),
               'log_rmse':float(np.sqrt(np.mean(errors**2))),
               'digitization_gmfe_bounds':bounded_gmfe(yp,lo,hi),
               'gmfe_excluding_unlisted_1_25h':float(np.exp(np.abs(errors[listed]).mean())),
               'compatibility':pred['compatibility'],
               'provenance':{'prediction':pred_ref,'job_id':job['job_id']}}
        case['curves'].append(curve)
        # Include horizontal digitization uncertainty for figure-only times.
        lower_errors=[]; upper_errors=[]
        for j,point in enumerate(obs['points']):
            dt=point['time_digitization_error_bound_h']
            start,end=ot[j]-dt,ot[j]+dt
            assert t[0]<=start<=end<=t[-1]
            grid=np.r_[start,t[(t>start)&(t<end)],end]
            vals=np.interp(grid,t,y)
            pmin,pmax=float(vals.min()),float(vals.max())
            lower_errors.append(max(math.log(lo[j]/pmax),math.log(pmin/hi[j]),0.))
            upper_errors.append(max(abs(math.log(pmin/hi[j])),abs(math.log(pmax/lo[j]))))
        curve['digitization_gmfe_bounds']=[math.exp(float(np.mean(lower_errors))),math.exp(float(np.mean(upper_errors)))]
    ranking_lock=read(RUN/'rankings/ranking_lock.json')
    bi=ranking_lock['query_ids'].index(SID)
    for mid,label,_ in METHODS:
        r=ranking_lock['methods'][mid]
        verify_refs([r])
        scores=np.load(ROOT/r['path'],allow_pickle=False)[0,bi]
        case['rankings'].append({'method_id':mid,'method':label,'positions':positions(
            scores,ranking_lock['candidate_ids'],candidates,case)})
    arrays=np.load(OWN/'predictions/seed17.npz',allow_pickle=False)
    for mid,label,array in [('pbpk_match_light','PBPK-Match','scores'),
                            ('pbpk_match_light_document','PBPK-Match (document)','base_scores')]:
        case['rankings'].append({'method_id':mid,'method':label,'outer_fold':folds[0],
            'positions':positions(arrays[array][0,qi],own_protocol['candidate_ids'],candidates,case)})
    manifest=read(COMP/'predictions/manifest.json')
    ci=manifest['query_ids'].index(SID)
    ca=np.load(COMP/'predictions/component_scores.npz',allow_pickle=False)
    assert int(ca['indices'][ci])==qi and manifest['candidate_ids']==own_protocol['candidate_ids']
    case['rankings'].append({'method_id':COMPONENT_METHOD[0],'method':COMPONENT_METHOD[1],
        'outer_fold':folds[0],'positions':positions(ca['scores'][ci],manifest['candidate_ids'],candidates,case)})
    metrics={c['candidate_id']:curve_errors(case,c) for c in case['curves']}
    rows=[score_position(case,r,p,metrics) for r in case['rankings'] for p in r['positions']]
    best=min(case['curves'],key=lambda c:c['gmfe'])
    baseline=next(r for r in case['rankings'] if r['method_id']=='pbpk_match_light')
    first=baseline['positions'][0]
    success=first['tie_group_size']==1 and first['members'][0]['candidate_id']==best['candidate_id']
    robust=all(best['digitization_gmfe_bounds'][1]<c['digitization_gmfe_bounds'][0]
               for c in case['curves'] if c is not best)
    report={'schema':'post-hoc-curve-case-evaluation-v1','case':case,'position_rows':rows,
       'selection_test':{'basic_pbpk_top1_has_lowest_curve_gmfe':success,
                         'best_identity_robust_to_digitization_intervals':robust,
                         'lowest_gmfe_candidate':best['label']},
       'planned_scenarios':[{'scenario_id':SID,'state':'evaluated'},
          {'scenario_id':'clinical:caffeine_sherman2022_solution_50mg','state':'not_scored',
           'reason':'Only four unambiguous late-time means available; fails requested complete-curve example.'}],
       'selection':protocol['selection'],'original_cohort_statistics_modified':False,
       'provenance':[ref(p) for p in [EXT/'protocol.json',EXT/'protocol_amendment_01.json',
                      observation_path,extraction_receipt,Path(__file__).resolve()]]}
    write(EXT/'evaluation.json',report)
    write(EXT/'data_profile.json',{'points':len(ot),'time_range_h':[float(ot[0]),float(ot[-1])],
        'positive_concentration_range_ng_ml':[float(oy.min()),float(oy.max())],
        'candidates':3,'units':'h, ng/mL','missing_values':False,
        'observation_type':'digitized group mean, not independent replicates',
        'within_case_dependency':'All candidates share the same observed time points',
        'source_methods_omits_figure_point':1.25,'parameter_fitting':False})
    print(json.dumps(report['selection_test'],indent=2))
    print([(c['label'],c['gmfe'],c['digitization_gmfe_bounds'],c['gmfe_excluding_unlisted_1_25h']) for c in case['curves']])
    if not success:
        print('Selection criterion failed; report retained; publication view not changed.')
        return
    old=load_figure_data()
    rosu=next(c for c in old['cases'] if c['compound']=='rosuvastatin')
    # The omitted document-only row is redundant only after exact identity checks.
    doc=next(r for r in case['rankings'] if r['method_id']=='pbpk_match_light_document')
    signature=lambda r:[{k:v for k,v in p.items() if k!='score'} for p in r['positions']]
    assert signature(baseline)==signature(doc)
    display_case=json.loads(json.dumps(case))
    display_case['rankings']=[r for r in display_case['rankings'] if r['method_id']!='pbpk_match_light_document']
    output={'schema':'retrospective-curve-illustration-v1','cases':[display_case,rosu],
        'component_view':{'displayed_methods':[r['method_id'] for r in display_case['rankings']]},
        'selection_test':report['selection_test'],'extension_report':ref(EXT/'evaluation.json'),
        'selection':protocol['selection'],'original_cohort_statistics_modified':False,
        'validation':{'candidate_curves':8,'unique_qualified_observations':len(ot)+len(rosu['observation']['time_h']),
                      'independent_publications':2,'scenarios':2,'outer_held_out_predictions_verified':True},
        'provenance':report['provenance']+[ref(ROOT/'results/data/pbpk_match_light_curves.json')]}
    write(DEST,output)
    print(DEST)


if __name__=='__main__':
    main()
