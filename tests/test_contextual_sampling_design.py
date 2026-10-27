import numpy as np
import pytest

from pbpk_kb.blueprint_policy import contextual_sampling_design as p


def test_features_use_visible_shape_and_preserve_units():
    t=np.array([.25, 1, 8.]); vt=np.array([2,3,4,5.]); y=np.array([8,5,3,2.])
    c=np.array([[12,10,1],[20,16,2],[8,5,.7]])
    x=np.zeros((3,8));g=np.array([12,20,8.])
    a=p.contextual_features(x,t,vt,y,y*.9,c,g)
    b=p.contextual_features(x,t,vt,y*1000,y*.9*1000,c*1000,g*1000)
    np.testing.assert_allclose(a,b,atol=1e-12)
    assert a.shape==(3,12) and not np.allclose(a[:,8:],0)
    with pytest.raises(ValueError):p.contextual_features(x,t,vt[:3],y[:3],y[:3],c,g)


def synthetic():
    designs=[];labels={}
    for i in range(4):
        key=f'd{i}';x=np.zeros((3,12));x[:,0]=[0,1,-1];x[:,11]=[0,i,-i]
        designs.append(dict(design_key=key,case_id=key,drug_study_group=f'g{i}',X=x,time_index=0))
        labels[key]=dict(cmax=[2,1,4],curve=[1,2,1])
    return designs,labels


def test_training_membership_and_exact_time_anchor():
    d,l=synthetic()
    class Guard(dict):
        def __getitem__(self,key):
            assert key!='d3', 'Held-out outcome read'
            return super().__getitem__(key)
    m=p.train(d[:3],Guard(l))
    assert m['full']['train_case_ids']==['d0','d1','d2']
    np.testing.assert_allclose(p.predict(m['full'],d[3]['X'],0)[0],0)
    a=p.choose(m,d[3]['X'],[1,2,3],0)
    assert a['selected_index']==0  # requested gain at point 1 damages curve
    b=p.choose(m,d[3]['X'],[1,2,3],0,curve_guard=False)
    assert b['selected_index']==1
    for f in m['deletions']:
        assert f['excluded_group'] not in f['model']['train_groups']


def test_ridge_context_changes_ranking_not_just_menu_offset():
    x=np.zeros((3,8));t=[.5,1,8]; vt=[2,3,4,5];c=np.ones((3,3));g=[1,2,3]
    a=p.contextual_features(x,t,vt,[8,5,3,2],[8,5,3,2],c,g)
    b=p.contextual_features(x,t,vt,[2,3,5,8],[2,3,5,8],c,g)
    assert not np.allclose(a-a[0],b-b[0])


def test_all_labels_retain_alignment_and_reject_non_errors():
    d,l=synthetic();l['d0']['cmax']=[.5,1,4]
    with pytest.raises(ValueError):p.train(d,l)
