#!/usr/bin/env python3
"""Outcome-blind frozen-head input ablations and conservative routing.

Inference consumes public inputs, saved embeddings and frozen heads only.
Outcomes enter a separate evaluation phase after predictions are saved.
Rules are post-development diagnostics, not independently validated policies.
"""
from __future__ import annotations
import argparse, copy, csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_component_model_integration as run
from pbpk_match_light_inputs import visible_inputs,load_cache
OLD=run.OLD
SOURCE=run.DEFAULT
OUT=ROOT/'experiments/model_recommendation/reviewer_revision_20261009/component_diagnostics'
INFERENCE_KEYS=('scores','base_scores','probabilities','selected_predicted_error','error_supported')
ABLATIONS=('components_replay','text_only','structure_only','structure_without_process_types','structure_without_enzymes','structure_without_transporters')

def input_guard(row, original, probabilities, scores):
    """No label, outcome, candidate error or drug-specific threshold is accepted."""
    audit=row['audit']; additions=audit.get('added_features',{})
    if row['status']!='augmented': return False,'no_component_input'
    specific={field:set(additions.get(field,[])) for field in ('enzymes','transporters','compartments')}
    if not any(specific.values()): return False,'generic_adme_only'
    context=audit.get('evidence_context',{})
    links=[r for r in audit.get('soft_requirements',[]) if r['field'] in specific and r['value'] in specific[r['field']]]
    support=[context[e] for r in links for e in r['evidence_ids'] if e in context]
    if not support or any(c.get('conflicting_fields') for c in support): return False,'absent_or_conflicting_context'
    # The same 0.5 Y-probability threshold used in the existing binary endpoint.
    tied=np.flatnonzero(scores==scores.max())
    if not len(tied) or float(np.min(probabilities[tied,0]))<.5: return False,'top1_applicability_uncertain'
    return True,'specific_mechanism_and_y_probability_at_least_half'

def infer():
    import torch
    from pbpk_kb.recommendation.pbpk_match_v2.data import FIELDS
    torch.set_num_threads(4)
    protocol=run.verify_baseline();ledger=run.rows(run.verify_bridge_inputs(SOURCE))
    visible,cache=visible_inputs(),load_cache()
    indices=np.asarray([r['global_index'] for r in ledger],np.int64)
    folds=run.exact_fold_membership(indices,protocol)
    with np.load(OLD/'predictions/seed17.npz',allow_pickle=False) as z:
        baseline={k:z[k][0,indices].copy() for k in INFERENCE_KEYS}
    with np.load(SOURCE/'predictions/component_scores.npz',allow_pickle=False) as z:
        augmented={k:z[k].copy() for k in INFERENCE_KEYS}
    with np.load(SOURCE/'cache/augmented_embeddings.npz',allow_pickle=False) as z:
        augvectors=z['query_embeddings'].copy()
    features=visible['condition_features'][0,indices].copy()
    augfeatures=features.copy();changed=[j for j,r in enumerate(ledger) if r['status']=='augmented']
    for j in changed:
        augfeatures[j]=run.soft_features(visible['query_records'][indices[j]],ledger[j]['augmented_query'],visible['candidate_records'],features[j])
    assert len(changed)==50
    originalvectors=cache['light_query_embeddings'][0,indices].copy()
    requested=run.read(ROOT/'experiments/model_recommendation/pbpk_match_v2/dataset.json').get('execution_requested_endpoints',{})
    allresults={'original':baseline,'components':augmented};replay=[]
    for variant in ABLATIONS:
        vectors=augvectors if variant in ('text_only','components_replay') else originalvectors
        pairfeatures=(features if variant=='text_only' else augfeatures).copy()
        if variant.startswith('structure_without_'):
            field=variant.removeprefix('structure_without_');cols=slice(FIELDS.index(field)*5,FIELDS.index(field)*5+5)
            pairfeatures[:,:,cols]=features[:,:,cols]
        result={k:v.copy() for k,v in baseline.items()}
        for fold in range(5):
            members=[j for j in changed if folds[int(indices[j])]==fold]
            if not members:continue
            records=[ledger[j]['augmented_query'] for j in members]
            got=run.infer_fold(vectors[members],pairfeatures[members],cache['light_candidate_embeddings'],records,
                [requested.get(str(indices[j])) for j in members],OLD/f'runs/seed17/fold{fold}/outer',device='cpu')
            for k in INFERENCE_KEYS:result[k][members]=got[k]
            if variant=='text_only':
                j=members[:1];r=[visible['query_records'][indices[i]] for i in j]
                same=run.infer_fold(originalvectors[j],features[j],cache['light_candidate_embeddings'],r,
                    [requested.get(str(indices[i])) for i in j],OLD/f'runs/seed17/fold{fold}/outer',device='cpu')
                maxdiff=max(float(np.max(np.abs(same[k].astype(float)-baseline[k][j].astype(float)))) for k in INFERENCE_KEYS)
                assert maxdiff<2e-5,(fold,maxdiff)
                replay.append({'fold':fold,'max_absolute_difference':maxdiff})
        if variant=='components_replay':
            maxdiff=max(float(np.max(np.abs(result[k].astype(float)-augmented[k].astype(float)))) for k in INFERENCE_KEYS)
            assert maxdiff<2e-5,('full_augmented_replay',maxdiff)
            for key in ('scores','base_scores'):
                assert np.array_equal(np.argsort(-result[key],kind='stable'),np.argsort(-augmented[key],kind='stable')),key
            replay.append({'kind':'all_186_augmented_rows','max_absolute_difference':maxdiff,'ranks_identical':True})
        else:
            allresults[variant]=result
    guarded={k:v.copy() for k,v in baseline.items()};routing=[]
    for j,row in enumerate(ledger):
        accepted,reason=input_guard(row,visible['query_records'][indices[j]],augmented['probabilities'][j],augmented['scores'][j])
        if accepted:
            for k in INFERENCE_KEYS:guarded[k][j]=augmented[k][j]
        routing.append({'query_id':row['query_id'],'cohort':row['cohort'],'compound':row['augmented_query']['compounds'][0],
                        'accept_component_input':accepted,'reason':reason})
    allresults['guarded_components']=guarded
    OUT.mkdir(parents=True,exist_ok=True)
    for name,values in allresults.items():run.array_write(OUT/f'{name}.npz',indices=indices,**values)
    run.write(OUT/'inference.json',{'status':'complete','purpose':'post_development_input_ablation',
        'outcomes_used_in_inference':False,'new_training':False,'new_llm':False,'queries':186,'candidates':280,
        'methods':list(allresults),'changed_input_requests':50,'routing':routing,'routing_reasons':dict(Counter(r['reason'] for r in routing)),
        'replay':replay,'sources':[run.ref(p) for p in [Path(__file__),SOURCE/'inputs/augmented_queries.jsonl',
             SOURCE/'predictions/component_scores.npz',SOURCE/'cache/augmented_embeddings.npz',OLD/'predictions/seed17.npz',OLD/'protocol.json']],
        'predictions':[run.ref(OUT/f'{n}.npz') for n in allresults],
        'guard_definition':'No new specific enzyme/transporter/compartment -> original input; explicit evidence context conflict -> original input; every tied first candidate must have frozen calibrated Y probability >= 0.5, else original input.',
        'guard_limit':'A conservative opt-in development check; source context can remain unspecified, and Y probability is automatic-label-derived, not clinical safety validation.'})

def evaluate():
    import evaluate_component_model_integration as ev
    receipt=run.read(OUT/'inference.json')
    for r in receipt['sources']+receipt['predictions']:run.verify_ref(r)
    data=ev.load_dataset();ledger=run.rows(SOURCE/'inputs/augmented_queries.jsonl')
    indices=np.asarray([r['global_index'] for r in ledger]);lookup={i:j for j,i in enumerate(indices)}
    core=[lookup[i] for i in data['core_indices']];clinical=[lookup[i] for i in data['execution_indices']]
    selection=run.read(ROOT/'experiments/model_recommendation/frontier_core_145_20260921/selection_manifest.json')
    binary=[j for j,i in enumerate(data['core_indices']) if selection['compound_split'][data['query_records'][i]['compound_id']]=='test']
    ids=[c['candidate_id'] for c in data['candidate_records']];qids=[data['query_records'][i]['query_id'] for i in data['execution_indices']]
    contract=ev.ev.clinical_contract();methods=[];details={};scores={};traces=[]
    for method in receipt['methods']:
        with np.load(OUT/f'{method}.npz',allow_pickle=False) as z: sc=z['scores'];pr=z['probabilities']
        scores[method]=sc
        q=ev.evaluate_quality(sc,pr,data,core,binary)
        c=ev.ev.clinical_evaluation(sc[clinical],qids,ids,method=method,contract=contract)
        c['omission_summary']=ev.light.omission_summary(c['omissions'])
        details[method]={'quality':q,'execution':c}
        methods.append({'method':method,'quality':q['judged_only'],'known_top10':q['known_top10_coverage'],
             'binary':q['paper_binary_102'],'cmax':[r for r in c['position_summary'] if r['endpoint']=='Cmax_reference_vs_group'],
             'omission_summary':c['omission_summary']})
    for j,row in enumerate(ledger):
        if row['cohort']!='execution' or row['augmented_query']['compounds']!=['digoxin']:continue
        result=run.read(row['component_result']['path']);candidates=data['candidate_records']
        candidate_indices=sorted(set(int(i) for method in scores for i in np.argsort(-scores[method][j])[:3]))
        candidates_trace=[]
        for i in candidate_indices:
            cid=ids[i];err=contract[row['query_id']]['errors'].get(cid,{}).get('Cmax_reference_vs_group',{}).get('absolute')
            candidates_trace.append({'candidate_id':cid,'candidate_text':candidates[i]['text'],'candidate_features':candidates[i].get('features'),
                'cmax_fold_error':float(np.exp(err)) if err is not None else None,
                'scores':{method:float(sc[j,i]) for method,sc in scores.items()},
                'ranks':{method:int((sc[j]>sc[j,i]).sum()+1) for method,sc in scores.items()}})
        traces.append({'query_id':row['query_id'],'original_text':row['original_text'],'augmented_text':row['augmented_query']['text'],
            'audit':row['audit'],'citations':result['recommendation']['citations'],'candidates':candidates_trace})
    payload={'status':'complete','interpretation':'Post-development diagnostic; all outcomes loaded only after immutable prediction receipt.',
        'summary':methods,'details':details,'digoxin_traces':traces,'inference':run.ref(OUT/'inference.json'),
        'outcome_source':run.ref(ev.ev.PRIMARY)}
    run.write(OUT/'evaluation.json',json.loads(json.dumps(payload,default=ev.json_default,allow_nan=False)))
    print(json.dumps({'methods':list(details),'routing':receipt['routing_reasons']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate-only',action='store_true');a=p.parse_args()
    if not a.evaluate_only:infer()
    evaluate()
