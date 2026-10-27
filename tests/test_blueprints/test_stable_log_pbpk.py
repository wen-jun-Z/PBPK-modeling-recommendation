"""Numerical identities and frozen-input regressions, without target outcomes."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from pbpk_kb.blueprint_policy import blood_partition_calibration as blood
from pbpk_kb.blueprint_policy import permeability_model as exchange
from pbpk_kb.blueprint_policy.stable_log_pbpk import predict_log, shifted_log_observation


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'experiments/blueprint_design/published_inference_controls_20261011/protocol.json'


def one_compartment(k=35., volume=4.):
    return SimpleNamespace(matrix=np.array([[-k, 0.], [k, 0.]]),
        input_vector=np.array([1., 0.]), observation_vector=np.array([1/volume, 0.]),
        state_names=('Central', 'Eliminated'))


@pytest.mark.parametrize('duration', [0., .5, 2.])
def test_analytic_bolus_and_infusion_survive_underflow(duration):
    k, volume, dose = 35., 4., 120.
    times = np.array([0., .1, .5, 2., 24., 100.])
    logs, diag = shifted_log_observation(one_compartment(k, volume), times, dose, duration)
    if duration == 0:
        expected = math.log(dose/volume)-k*times
    else:
        expected = []
        for t in times:
            expected.append(-np.inf if t == 0 else
                math.log(dose/(duration*k*volume)) + math.log(-math.expm1(-k*min(t,duration)))
                - k*max(0.,t-duration))
    np.testing.assert_allclose(logs, expected, rtol=1e-12, atol=1e-11)
    assert logs[-1] < -3000
    assert diag['matrix_unchanged'] is True
    assert diag['matrix_mass_residual'] == 0


def test_defective_eigensystem_erlang_transit_analytic():
    # Repeated eigenvalues have no invertible eigenvector basis. The exact
    # downstream response is dose*k*t*exp(-k*t), including subnormal tails.
    k = 40.
    system = SimpleNamespace(matrix=np.array([[-k,0.,0.],[k,-k,0.],[0.,k,0.]]),
        input_vector=np.array([1.,0.,0.]), observation_vector=np.array([0.,1.,0.]),
        state_names=('Input','Transit','Eliminated'))
    times=np.array([0.,.01,1.,24.])
    logs,_=shifted_log_observation(system,times,10.,0.)
    assert np.isneginf(logs[0])
    np.testing.assert_allclose(logs[1:],np.log(10*k*times[1:])-k*times[1:],atol=1e-11)


def test_invalid_nonconserving_matrix_rejected():
    system=one_compartment()
    system.matrix[1,0]-=1
    with pytest.raises(ValueError,match='conserve'):
        shifted_log_observation(system,[1.,2.],10.,.5)


def test_absorbing_state_cannot_reenter_body():
    system=one_compartment()
    system.matrix[:,1]=[1.,-1.]
    with pytest.raises(ValueError,match='absorbing'):
        shifted_log_observation(system,[1.,2.],10.,.5)


def test_zero_dose_is_zero_exposure_not_imputed_floor():
    logs,_=shifted_log_observation(one_compartment(),[0.,1.,24.],0.,.5)
    assert np.isneginf(logs).all()


def protocol():
    path=PROTOCOL
    if not path.is_file():
        path=PROTOCOL.parent/'revisions/numerical_underflow/protocol.json'
    if not path.is_file():
        pytest.skip('Frozen retrospective inference input protocol unavailable')
    return json.loads(path.read_text())


def readref(value):
    return json.loads((ROOT/value['path']).read_text())


def native_input(p, cid, formula):
    j=p['jobs'][cid+'__snpe_c']['adapter']['formulas'][formula]
    return j,readref(j['development_physiology']),j['development_request']


def test_ertapenem_underflow_matches_independent_normalized_expm():
    p=protocol()
    j,phys,req=native_input(p,'ertapenem_pletz2004_day1_iv','PKSimStandard')
    x=np.array([4.6569549354517985,-1.9684067819342268,3.2121067229811473,2.4204502470926124])
    times=np.array(j['times_h'])
    old=blood.simulate(phys,req,times,x)
    assert old['concentration_mg_l'][-1] == 0
    logs,diag=predict_log(phys,req,times,x,return_diagnostics=True)
    assert diag['method']=='shifted_active_expm'
    assert logs[-1]==pytest.approx(-767.9193340527,abs=1e-7)
    np.testing.assert_allclose(logs[:-1],np.log(old['concentration_mg_l'][:-1]),atol=1e-7)
    assert old['mass_balance_relative_error'] < 1e-10

    derived=blood.derive_physiology(phys,req,x[3]);cl=blood.clearance_inputs(req)
    system=exchange.build_system(derived,total_clearance_l_h=cl['total_clearance_l_h'],
        renal_clearance_l_h=cl['renal_clearance_l_h'],
        log_parameters=dict(zip(blood.PARAMETERS[:3],x[:3])),observation='auto')
    # Independent full, unshifted generator includes the eliminated mass.
    dose=req['dosing']['dose_mg'];duration=req['dosing']['infusion_minutes']/60
    full=system.matrix
    augmented=np.zeros((len(full)+1,len(full)+1))
    augmented[:-1,:-1]=full;augmented[:-1,-1]=dose/duration*system.input_vector
    start=np.zeros(len(full)+1);start[-1]=1.
    end=(expm(augmented*duration)@start)[:-1]
    assert end.sum()==pytest.approx(dose,rel=1e-10)
    active=full[:-1,:-1];observer=system.observation_vector[:-1]
    tau=float(times[-1]-duration);nstep=math.ceil(tau)
    transition=expm(active*(tau/nstep));amount=end[:-1].copy();scale=0.
    for _ in range(nstep):
        amount=transition@amount
        norm=float(amount.max());assert norm > 0
        amount/=norm;scale+=math.log(norm)
    independent=math.log(float(observer@amount))+scale
    assert logs[-1]==pytest.approx(independent,abs=2e-8)


def test_thirty_frozen_development_predictions_same_model():
    p=protocol()
    assert len(p['cases'])==30
    counts={'frozen_solver_exact':0,'shifted_active_expm':0}
    for cid,case in p['cases'].items():
        unit=readref(case['frozen_pbpk_prediction'])
        formula=unit['formula'];x=np.array(unit['parameters']['log_parameters'])
        j,phys,req=native_input(p,cid,formula)
        times=np.array(j['times_h'])
        old=blood.simulate(phys,req,times,x)
        logs,diag=predict_log(phys,req,times,x,return_diagnostics=True)
        expected=np.log(old['concentration_mg_l'])
        if diag['method']=='frozen_solver_exact':
            np.testing.assert_array_equal(logs,expected)
        else:
            np.testing.assert_allclose(logs,expected,rtol=1e-9,atol=1e-7)
        assert old['mass_balance_relative_error'] < 1e-8
        counts[diag['method']]+=1
    assert sum(counts.values())==30


def test_thirty_locked_prediction_grids_without_observed_outcomes():
    p=protocol()
    for cid,case in p['cases'].items():
        unit=readref(case['frozen_pbpk_prediction'])
        formula=unit['formula'];x=np.array(unit['parameters']['log_parameters'])
        inp=p['jobs'][cid+'__snpe_c']['adapter']['formulas'][formula]
        # Only the already locked model's time grid and administration request
        # are read. No observed concentration or prediction error is accessed.
        curve=readref(unit['prediction'])
        times=np.array(curve['time_min'])/60
        phys=readref(inp['target_physiology']);req=inp['target_request']
        original=blood.simulate(phys,req,times,x)
        logs=predict_log(phys,req,times,x)
        with np.errstate(under='ignore'):
            updated=np.exp(logs)
        expected=original['concentration_mg_l']
        np.testing.assert_allclose(updated,expected,rtol=1e-8,
            atol=1e-10*max(float(np.max(expected)),1e-30))
        assert original['mass_balance_relative_error'] < 1e-8
