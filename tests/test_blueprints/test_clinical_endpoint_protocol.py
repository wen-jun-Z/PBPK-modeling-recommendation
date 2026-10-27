"""Guard study-specific clinical comparisons against mismatched NCA windows."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_clinical_method_comparison import predicted_nca, reporting_curve_indices
from pbpk_kb.blueprints.clinical_prediction_nca import compute_clinical_prediction_nca


@pytest.fixture
def prediction():
    t = np.linspace(0, 20, 2001)
    c = np.where(t <= 1, 10*(1-np.exp(-t)), 10*(1-math.exp(-1))*np.exp(-(t-1)))
    return dict(time_h=t.tolist(), concentration_ng_ml=c.tolist(), dose_ng=1000.,
                infusion_h=1., nominal_times_h=np.arange(0, 12.01, .25).tolist(),
                lloq_ng_ml=.2, weight_kg=70.)


def test_unextended_policy_preserves_existing_nca_exactly(prediction):
    assert predicted_nca(policy={}, **prediction) == compute_clinical_prediction_nca(**prediction)


def test_actual_clinical_collection_after_nominal_nca_time_is_covered():
    times = [0., 4., 5., 5.2, 5.3, 6.]
    policy = {'sampling_time_h': [0., 4., 5.]}
    assert reporting_curve_indices(times, policy, {'window_end_h': 5.233333}) == [0, 1, 2, 3, 4]
    assert policy['sampling_time_h'][-1] == 5.
    with pytest.raises(ValueError):
        reporting_curve_indices(times[:4], policy, {'window_end_h': 5.233333})


def test_fixed_time_is_not_replaced_by_last_quantifiable_time(prediction):
    policy = dict(sampling_time_h=prediction['nominal_times_h'], c_last_time_h=8.)
    baseline = predicted_nca(policy={}, **prediction)
    actual = predicted_nca(policy=policy, **prediction)
    assert baseline['diagnostics']['tlast_h'] < 8
    assert actual['endpoints']['c_last']['value'] == pytest.approx(10*(1-math.exp(-1))*math.exp(-7))
    assert actual['endpoints']['c_last']['value'] < prediction['lloq_ng_ml']
    for key in ['auc_inf', 'terminal_half_life', 'clearance', 'vz']:
        assert actual['endpoints'][key] == baseline['endpoints'][key]


def test_fixed_auc_matches_declared_window_not_quantifiable_tail(prediction):
    policy = dict(sampling_time_h=prediction['nominal_times_h'], auc_window_end_h=2., integration='linear_trapezoidal')
    baseline = predicted_nca(policy={'integration':'linear_trapezoidal'}, **prediction)
    actual = predicted_nca(policy=policy, **prediction)
    t = np.arange(0, 2.01, .25)
    c = np.interp(t,prediction['time_h'],prediction['concentration_ng_ml'])
    expected = sum((b-a)*(x+y)/2 for a,b,x,y in zip(t,t[1:],c,c[1:]))
    assert actual['endpoints']['auc']['value'] == pytest.approx(expected)
    assert actual['endpoints']['auc_inf'] == baseline['endpoints']['auc_inf']
    assert actual['endpoints']['auc']['value'] != baseline['endpoints']['auc']['value']


def test_original_uncorrected_mrt_convention_is_explicit(prediction):
    baseline = predicted_nca(policy={}, **prediction)
    actual = predicted_nca(policy={'mrt_infusion_correction':'none'}, **prediction)
    assert actual['endpoints']['mrt']['value'] == pytest.approx(baseline['endpoints']['mrt']['value']+.5)
    assert actual['endpoints']['vss']['value'] == pytest.approx(
        actual['endpoints']['mrt']['value']*actual['endpoints']['clearance']['value'])
    assert actual['endpoints']['auc_inf'] == baseline['endpoints']['auc_inf']


@pytest.mark.parametrize('end',[True,-1,0,float('nan'),float('inf'),13,21])
def test_invalid_or_outside_protocol_window_rejected(prediction,end):
    with pytest.raises(ValueError):
        predicted_nca(policy={'sampling_time_h':prediction['nominal_times_h'], 'auc_window_end_h':end}, **prediction)
