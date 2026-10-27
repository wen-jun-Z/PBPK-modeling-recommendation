"""Selection must not disguise worse extreme-CL behavior as improvement."""
import numpy as np
import pytest

from pbpk_kb.blueprint_learning import clearance_similarity as cl


def test_middle_only_gain_cannot_override_extreme_regression():
    y=np.array([-2.,-.5,.2,.5,.8,1.,2.,3.])
    base=y+.5
    adverse=y.copy();adverse[:2]+=.6;adverse[-2:]+=.6
    result=cl.evaluate_candidates(y,{cl.BASE:base,'middle_only':adverse},np.arange(len(y))%3)
    assert result['scores']['middle_only']['log10_mae']<result['scores'][cl.BASE]['log10_mae']
    assert result['selected']==cl.BASE
    assert not result['scores']['middle_only']['eligibility_checks']['low_cl_not_worse']
    assert not result['scores']['middle_only']['eligibility_checks']['high_cl_not_worse']


def test_coherent_improvement_is_eligible_and_tie_retains_original():
    y=np.array([-2.,-.5,.2,.5,.8,1.,2.,3.]);folds=np.arange(len(y))%3
    selected=cl.evaluate_candidates(y,{cl.BASE:y+.5,'better':y+.3},folds)
    assert selected['selected']=='better'
    tied=cl.evaluate_candidates(y,{cl.BASE:y+.5,'equal':y+.5},folds)
    assert tied['selected']==cl.BASE


def test_tail_weights_are_bounded_and_respect_equal_frequency_groups():
    assert np.allclose(cl.tail_weights([-2,-1,.3,.5,2,3]),np.ones(6))
    y=np.array([-2.,.1,.2,.3,.4,.5,.6,2.])
    w=cl.tail_weights(y)
    assert w[0]>w[1] and w[-1]>w[1] and np.all(w>0) and np.all(w<=3)
    assert np.allclose(w,cl.tail_weights(y.copy()))
    with pytest.raises(ValueError):cl.tail_weights([float('nan')])


def test_nonfinite_oof_and_missing_tail_group_are_rejected():
    y=np.array([-2.,.2,2.]);folds=np.arange(3)
    with pytest.raises(ValueError):cl.evaluate_candidates(y,{cl.BASE:[-1,.1,float('nan')]},folds)
    with pytest.raises(ValueError):cl.evaluate_candidates([0.,.2,.3],{cl.BASE:[0,.2,.3]},folds)
