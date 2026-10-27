from copy import deepcopy
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from reviewer_component_diagnostics import input_guard
from reviewer_graph_diagnostics import paired_cluster
from reviewer_component_safe_rank import safe_ranking

def example():
    return {'status':'augmented','audit':{
        'added_features':{'enzymes':['cyp3a4']},
        'soft_requirements':[{'field':'enzymes','value':'cyp3a4','evidence_ids':['e1']}],
        'evidence_context':{'e1':{'conflicting_fields':[]}}}}

def test_generic_adme_does_not_gain_authority_from_probability():
    row=example();row['audit']['added_features']={'process_types':['absorption','distribution']}
    assert input_guard(row,{},np.array([[1.,0.,0.]]),np.array([1.]))==(False,'generic_adme_only')

def test_specific_mechanism_still_requires_all_top_ties_known_applicable():
    row=example()
    assert input_guard(row,{},np.array([[.6,.2,.2],[.49,.1,.41]]),np.array([1.,1.]))[0] is False
    assert input_guard(row,{},np.array([[.6,.2,.2],[.51,.1,.39]]),np.array([1.,1.]))[0] is True

def test_conflict_causes_fallback_without_any_outcome():
    row=example();row['audit']['evidence_context']['e1']['conflicting_fields']=['species']
    assert input_guard(row,{},np.array([[.99,.01,0.]]),np.array([1.]))==(False,'absent_or_conflicting_context')

def test_guard_ignores_unrelated_outcomes_and_does_not_mutate_input():
    row=example();original=deepcopy(row)
    expected=input_guard(row,{},np.array([[.6,.2,.2]]),np.array([1.]))
    row['actual_GMFE']=1e9;row['label']='N'
    assert input_guard(row,{},np.array([[.6,.2,.2]]),np.array([1.]))==expected
    row.pop('actual_GMFE');row.pop('label');assert row==original

def test_cluster_resampling_counts_correlated_queries_as_one_group():
    result=paired_cluster({'a':0.,'b':0.},{'a':1.,'b':1.},{'a':'shared','b':'shared'},replicates=100)
    assert result['queries']==2 and result['semantic_groups']==1
    assert result['exact_group_sign_flip_p']==1.
    assert result['ci95']==[1.,1.]

def test_safe_wrapper_defaults_to_original_document_and_retains_candidates():
    original={'query_id':'q','candidate_ids':['a','b'],'base_scores':np.array([2.,1.]),'scores':np.array([0.,1.]),
              'probabilities':np.array([[.6,.1,.3],[.7,.1,.2]])}
    out=safe_ranking(original)
    assert [r['candidate_id'] for r in out['ranking']]==['a','b']
    assert not out['routing']['error_correction_enabled']
    out=safe_ranking(original,enable_error_correction=True)
    assert [r['candidate_id'] for r in out['ranking']]==['b','a']

def test_safe_wrapper_rejects_cross_request_and_candidate_misalignment():
    import pytest
    original={'query_id':'q','candidate_ids':['a'],'base_scores':np.array([2.]),'scores':np.array([0.]),'probabilities':np.array([[.6,.1,.3]])}
    row=example();row['query_id']='q'
    component={**original,'query_id':'other'}
    with pytest.raises(ValueError,match='another request'):safe_ranking(original,component,row,enable_components=True)
    component={**original,'candidate_ids':['b']}
    with pytest.raises(ValueError,match='alignment'):safe_ranking(original,component,row,enable_components=True)
