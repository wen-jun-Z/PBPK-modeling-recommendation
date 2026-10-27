#!/usr/bin/env python3
"""Explicit opt-in wrapper for already computed full-catalog score vectors.

Default: original-request document score. Neither soft components nor predicted
error correction is activated implicitly. No evaluation label or PK observation
is accepted. For genuinely new requests, supply the corresponding scorer output
to ``safe_ranking``; the CLI replays one existing, source-bound study request.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import reviewer_component_diagnostics as diag

def safe_ranking(original,component=None,component_record=None,*,enable_components=False,enable_error_correction=False):
    """Route full vectors using identity, source context and frozen probabilities.

    Each prediction must carry query_id, candidate_ids, base_scores, scores and
    probabilities (Y,N,U). No score is re-fit and no candidate is dropped.
    The caller retains responsibility for generating source-bound model inputs.
    """
    def validate(p):
        ids=p['candidate_ids'];size=len(ids)
        if not p.get('query_id') or size==0 or len(set(ids))!=size:raise ValueError('Missing request identity or duplicate candidates')
        for key in ('scores','base_scores'):
            if np.asarray(p[key]).shape!=(size,) or not np.isfinite(p[key]).all():raise ValueError('Incomplete scores')
        prob=np.asarray(p['probabilities'])
        if prob.shape!=(size,3) or not np.isfinite(prob).all() or (prob<0).any() or not np.allclose(prob.sum(1),1,atol=1e-6):
            raise ValueError('Invalid applicability probabilities')
    validate(original)
    selected=original;accepted=False;reason='components_disabled'
    score_key='scores' if enable_error_correction else 'base_scores'
    if enable_components:
        if component is None or component_record is None:reason='no_component_prediction'
        else:
            validate(component)
            if component['query_id']!=original['query_id'] or component_record['query_id']!=original['query_id']:
                raise ValueError('Component result belongs to another request')
            if component['candidate_ids']!=original['candidate_ids']:raise ValueError('Candidate alignment differs')
            accepted,reason=diag.input_guard(component_record,{},np.asarray(component['probabilities']),np.asarray(component[score_key]))
            if accepted:selected=component
    scores=np.asarray(selected[score_key],float)
    levels=sorted(set(scores),reverse=True)
    ranks={value:1+int((scores>value).sum()) for value in levels}
    ordering=sorted(range(len(scores)),key=lambda i:(-scores[i],original['candidate_ids'][i]))
    return {'query_id':original['query_id'],'routing':{'components_requested':enable_components,'components_accepted':accepted,
        'reason':reason,'error_correction_enabled':enable_error_correction,
        'qualification':'development-only guard, not clinical applicability approval'},
        'ranking':[{'candidate_id':original['candidate_ids'][i],'rank':ranks[scores[i]],'score':float(scores[i]),
                    'probabilities_Y_N_U':np.asarray(selected['probabilities'][i],float).tolist()} for i in ordering]}

def replay(query_id,components=False,error_correction=False):
    manifest=diag.run.read(diag.SOURCE/'predictions/manifest.json')
    diag.run.verify_baseline()
    if diag.run.sha(diag.SOURCE/'predictions/component_scores.npz')!=manifest['predictions_sha256']:raise ValueError('Component score receipt differs')
    ledger=diag.run.rows(diag.run.verify_bridge_inputs(diag.SOURCE));j=manifest['query_ids'].index(query_id)
    if ledger[j]['query_id']!=query_id:raise ValueError('Ledger ordering differs')
    ids=manifest['candidate_ids'];gi=ledger[j]['global_index']
    with np.load(diag.OLD/'predictions/seed17.npz',allow_pickle=False) as z:
        original={k:z[k][0,gi] for k in ('scores','base_scores','probabilities')}
    with np.load(diag.SOURCE/'predictions/component_scores.npz',allow_pickle=False) as z:
        component={k:z[k][j] for k in ('scores','base_scores','probabilities')}
    for p in (original,component):p.update(query_id=query_id,candidate_ids=ids)
    return safe_ranking(original,component,ledger[j],enable_components=components,enable_error_correction=error_correction)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--query-id',required=True)
    p.add_argument('--enable-components',action='store_true');p.add_argument('--enable-error-correction',action='store_true')
    a=p.parse_args();print(json.dumps(replay(a.query_id,a.enable_components,a.enable_error_correction),ensure_ascii=False,indent=2))
