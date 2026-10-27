"""Coordinate profiles checked against an analytic log-linear model."""
from types import SimpleNamespace

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import multiaxis_evidence_profile as profile


def design(times):
    columns=np.column_stack([np.ones(len(times)),np.sin(times),np.cos(2*times),np.sin(3*times)])
    return np.linalg.qr(columns)[0]


def problem(n=12):
    times=np.linspace(.1,5.,n)
    truth=np.array([.2,-.1,.3,.6])
    observed=np.exp(design(times)@truth)
    def predict(x,t):return {'concentration_mg_l':np.exp(design(t)@x)}
    return predict,times,observed,np.full(4,-2.),np.full(4,2.),truth


def test_all_axes_match_analytic_profile_in_original_parameter_order():
    pred,t,y,lower,upper,truth=problem()
    result=profile.analyze(pred,t,y,lower,upper,grid_points=5,max_calls=5000)
    assert result['status']=='profiled' and result['failures']==0
    assert {r['axis'] for r in result['profile']}=={0,1,2,3}
    for row in result['profile']:
        axis=row['axis'];expected=truth.copy();expected[axis]=row['fixed_coordinate']
        actual=np.array(row['best']['log_parameters'])
        np.testing.assert_allclose(actual,expected,atol=1e-6)
        assert actual[axis]==row['fixed_coordinate']
        assert row['mse']==pytest.approx((row['fixed_coordinate']-truth[axis])**2/len(t),abs=1e-10)
        assert row['mse']==row['best']['mse']
        assert len(row['starts'])==2
        assert all(len(s['free_axes'])==3 and axis not in s['free_axes'] for s in row['starts'])
    assert result['calls']==result['full_fit']['calls']+sum(r['calls'] for r in result['profile'])
    assert result['calls']<=5000
    assert result['target_observations_used'] is False
    assert result['external_warm_start_used'] is False
    assert 'not a joint posterior' in result['interpretation']


def test_grid_has_bounds_native_zero_and_visible_only_location():
    pred,t,y,lower,upper,truth=problem()
    result=profile.analyze(pred,t,y,lower,upper,grid_points=4,max_calls=5000)
    for axis,name in enumerate(profile.AXES):
        grid=result['axis_grids'][name]
        assert min(grid)==lower[axis] and max(grid)==upper[axis] and 0. in grid
        assert any(abs(v-truth[axis])<1e-6 for v in grid)


def test_no_cross_node_or_location_fit_warm_start():
    pred,t,y,lower,upper,_=problem()
    result=profile.analyze(pred,t,y,lower,upper,grid_points=3,max_calls=5000)
    independent=np.random.default_rng(profile.SEED).uniform(lower,upper)
    for row in result['profile']:
        axis=row['axis'];free=[i for i in range(4) if i!=axis]
        np.testing.assert_array_equal(np.array(row['starts'][0]['initial_log_parameters'])[free],np.zeros(3))
        np.testing.assert_array_equal(np.array(row['starts'][1]['initial_log_parameters'])[free],independent[free])


def test_guard_is_before_every_charged_call_and_failure_propagates():
    pred,t,y,lower,upper,_=problem();events=[]
    def guard():
        events.append('guard')
        if len(events)==5:raise ValueError('Changed source hash')
    def predict(x,t):events.append('predict');return pred(x,t)
    with pytest.raises(ValueError,match='Changed source hash'):
        profile.analyze(predict,t,y,lower,upper,before_trial=guard)
    assert events==['guard','predict','guard','predict','guard']


def test_hard_budget_counts_failures_and_retains_every_axis_node(monkeypatch):
    def hungry(fun,initial,**kwargs):
        for _ in range(10000):fun(initial)
        pytest.fail('Budget was not enforced')
    monkeypatch.setattr(profile,'least_squares',hungry)
    _,t,y,lower,upper,_=problem();calls=[]
    def invalid(*args):calls.append(1);raise ValueError('Synthetic numerical failure')
    result=profile.analyze(invalid,t,y,lower,upper,grid_points=3,max_calls=37)
    assert result['status']=='incomplete' and result['calls']==result['failures']==len(calls)==37
    assert len(result['profile'])==12 and {r['axis'] for r in result['profile']}==set(range(4))
    assert result['full_fit']['calls']+sum(r['calls'] for r in result['profile'])==37
    assert all(r['best'] is None and not r['successful'] for r in result['profile'])


def test_optimizer_limit_is_not_eligible_even_with_finite_best(monkeypatch):
    def truncated(fun,initial,**kwargs):fun(initial);return SimpleNamespace(success=False)
    monkeypatch.setattr(profile,'least_squares',truncated)
    pred,t,y,lower,upper,_=problem()
    result=profile.analyze(pred,t,y,lower,upper,grid_points=3,max_calls=100)
    assert result['status']=='incomplete'
    for row in result['profile']:
        assert row['best'] is None and not row['successful']
        assert all(s['best'] is not None and s['status']=='optimizer_limit' for s in row['starts'])


def test_small_budget_has_explicit_unattempted_records():
    pred,t,y,lower,upper,_=problem()
    result=profile.analyze(pred,t,y,lower,upper,grid_points=3,max_calls=2)
    assert result['calls']==2 and len(result['profile'])==12 and result['status']=='incomplete'
    assert any(s['status']=='no_call_quota' for r in result['profile'] for s in r['starts'])


def test_inputs_are_unchanged_and_predictor_receives_copies():
    pred,t,y,lower,upper,_=problem();originals=[a.copy() for a in (t,y,lower,upper)]
    def mutating(x,times):
        answer=pred(x,times);x[:]=999.;times[:]=999.;return answer
    result=profile.analyze(mutating,t,y,lower,upper,grid_points=3,max_calls=5000)
    assert result['status']=='profiled'
    for current,old in zip((t,y,lower,upper),originals):np.testing.assert_array_equal(current,old)


def test_grid_can_profile_bounds_without_native_zero():
    pred,t,y,lower,upper,_=problem();lower[0]=.1
    result=profile.analyze(pred,t,y,lower,upper,grid_points=3,max_calls=5000)
    assert result['axis_grids'][profile.AXES[0]][0]==.1
    assert 0. not in result['axis_grids'][profile.AXES[0]]
    for row in result['profile']:
        if row['best'] is not None:assert row['best']['log_parameters'][0]>=.1


@pytest.mark.parametrize('name,value',[('grid_points',2),('grid_points',True),('grid_points',3.5),('max_calls',0),('max_calls',False),('max_calls',2.5)])
def test_bad_options_fail_before_prediction(name,value):
    _,t,y,lower,upper,_=problem()
    with pytest.raises(ValueError):
        profile.analyze(lambda *a:pytest.fail('Invalid request predicted'),t,y,lower,upper,**{name:value})


@pytest.mark.parametrize('fault',['three_points','duplicate_times','negative_time','zero_concentration','bad_bounds','nonfinite_bound'])
def test_invalid_visible_data_rejected(fault):
    _,t,y,lower,upper,_=problem()
    if fault=='three_points':t,y=t[:3],y[:3]
    elif fault=='duplicate_times':t[1]=t[0]
    elif fault=='negative_time':t[0]=-1.
    elif fault=='zero_concentration':y[0]=0.
    elif fault=='bad_bounds':upper[0]=lower[0]
    else:upper[1]=np.inf
    with pytest.raises(ValueError):profile.analyze(lambda *a:pytest.fail('Invalid data predicted'),t,y,lower,upper)
