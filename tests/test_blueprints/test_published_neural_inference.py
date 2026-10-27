"""Statistical and isolation checks for the official SNPE-C adapter."""
import math

import numpy as np
import pytest
from scipy.stats import truncnorm

from pbpk_kb.blueprint_policy.published_neural_inference import EvidencePrior, fit


def test_prior_density_matches_independent_reference():
    lower=np.array([-5.,-2.,-4.,-3.])
    upper=np.array([4.,2.,4.,6.])
    sd=np.array([3.,6.,3.])
    p=EvidencePrior(lower,upper,sd)
    points=np.array([[.1,.2,.3,.4],[-4.,-1.9,3.8,5.2]])
    expected=-math.log(9)+sum(truncnorm.logpdf(points[:,i+1],lower[i+1]/sd[i],upper[i+1]/sd[i],scale=sd[i]) for i in range(3))
    np.testing.assert_allclose(p.log_prob(points).numpy(),expected,rtol=1e-6)
    assert np.isneginf(float(p.log_prob([5.,0.,0.,0.])))
    samples=p.sample((10000,)).numpy()
    assert np.all(samples>=lower) and np.all(samples<=upper)
    np.testing.assert_allclose(np.mean(samples,axis=0),p.mean.numpy(),atol=.12)


def test_prior_log_ratios_match_current_blueprint_penalty():
    n=5;sigma=.3
    sd=sigma*np.array([math.log(10),math.log(100),math.log(10)])/(math.sqrt(n)*.1)
    p=EvidencePrior([-8,-2,-4,-3],[8,2,4,6],sd)
    x=np.array([2.,.8,-1.5,1.2])
    scaled=-2*sigma**2/n*float(p.log_prob(x)-p.log_prob(np.zeros(4)))
    expected=.01*sum((x[i+1]/d)**2 for i,d in enumerate([math.log(10),math.log(100),math.log(10)]))
    assert scaled==pytest.approx(expected,abs=1e-7)


def test_invalid_visible_data_never_calls_simulator():
    calls=[]
    with pytest.raises(ValueError,match='finite visible'):
        fit(lambda x:calls.append(x),[1.,2.,np.nan,4.],np.zeros(4),[-2]*4,[2]*4,device='cpu')
    assert not calls


def test_failed_simulation_is_not_filtered():
    with pytest.raises(ValueError,match='filtering would bias'):
        fit(lambda x:np.full(5,np.nan),np.zeros(5),np.zeros(4),[-2]*4,[2]*4,
            device='cpu',rounds=1,simulations_per_round=32,max_epochs=1)


def test_infrastructure_pause_propagates():
    def stopped(x):
        raise RuntimeError('Protocol paused: source hash changed')
    with pytest.raises(RuntimeError,match='source hash changed'):
        fit(stopped,np.zeros(5),np.zeros(4),[-2]*4,[2]*4,
            device='cpu',rounds=1,simulations_per_round=32,max_epochs=1)


def test_official_sequential_correction_and_linear_posterior():
    # Known linear-Gaussian inverse task: no clinical target files are inputs.
    observed=np.array([.25,-.1,.35,.15,.15])
    result=fit(lambda x:np.r_[x,x[0]+x[1]],observed,np.zeros(4),[-2]*4,[2]*4,
        seed=20261011,device='cpu',rounds=2,simulations_per_round=2048,
        posterior_draws=1024,max_epochs=100,stop_after_epochs=15,
        hidden_features=32,num_transforms=3,training_batch_size=128)
    point=np.array(result['point_parameters'])
    np.testing.assert_allclose(point,observed[:4],atol=.25)
    assert result['simulator_calls']==4160
    assert result['diagnostics']['rounds'][1]['proposal']=='previous_round_corrected_posterior'
    assert result['diagnostics']['target_observations_used'] is False
    assert result['diagnostics']['backend_version']=='0.27.0'
