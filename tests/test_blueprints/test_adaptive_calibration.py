"""Objective and concurrent namespace contracts for alternative PBPK calibration."""
import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from pbpk_kb.blueprint_policy import adaptive_calibration as a
from pbpk_kb.blueprint_policy import native_calibration as baseline


def inputs(n=5):
    observed={'observations':[{'time_min':float(i+1),'concentration_umol_per_l':1.} for i in range(n)]}
    curve={'time_min':[float(i) for i in range(n+2)],'concentration':[2.]*(n+2),'unit':'umol/l','time_unit':'min'}
    return {},curve,observed,2.,2.,1.,.1


def test_squared_objective_matches_frozen_native_objective():
    args=inputs();r=a.score_native(*args,a.objective_config('trust_region',5))
    expected=math.log(2)**2+.1+.01*(1/2.5)**2
    assert r['objective']==pytest.approx(expected)
    assert np.dot(r['residual_vector'],r['residual_vector'])==pytest.approx(expected)


def test_sparse_prior_increases_only_prior_not_observation_loss():
    args=inputs();base=a.score_native(*args,a.objective_config('trust_region',5))
    sparse=a.score_native(*args,a.objective_config('trust_region_sparse',5))
    assert sparse['objective_components']['development_curve_loss']==base['objective_components']['development_curve_loss']
    for key in ('clearance_input_prior','lipophilicity_prior'):
        assert sparse['objective_components'][key]==pytest.approx(2*base['objective_components'][key])
    assert a.objective_config('trust_region_sparse',20)['prior_multiplier']==1.


def test_robust_data_loss_does_not_downweight_parameter_prior():
    args=list(inputs());args[1]['concentration']=[100.] * len(args[1]['concentration'])
    base=a.score_native(*args,a.objective_config('trust_region',5))
    robust=a.score_native(*args,a.objective_config('trust_region_robust',5))
    assert robust['objective']<base['objective']
    for key in ('clearance_input_prior','lipophilicity_prior'):
        assert robust['objective_components'][key]==base['objective_components'][key]
    assert robust['objective']==pytest.approx(sum(v*v for v in robust['residual_vector']))


def test_private_engine_namespaces_do_not_cross_contaminate_threads():
    original_version=baseline.VERSION;original_score=baseline._score
    def score(variant):
        engine=a._engine(a.objective_config(variant,5))
        assert engine.VERSION==a.VERSION
        return engine._score(*inputs())['objective']
    variants=['trust_region','trust_region_sparse','trust_region_robust']*4
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(score,variants))
    for i in range(3):assert len(set(results[i::3]))==1
    assert len(set(results[:3]))==3
    assert baseline.VERSION==original_version and baseline._score is original_score


def test_wrong_point_count_or_prior_rejected():
    with pytest.raises(ValueError,match='Observation count'):
        a.score_native(*inputs(),a.objective_config('trust_region',3))
    args=list(inputs());args[-1]=.2
    with pytest.raises(ValueError,match='weight 0.1'):
        a.score_native(*args,a.objective_config('trust_region',5))
