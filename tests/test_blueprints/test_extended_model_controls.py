"""Independent three-compartment numerics and frozen five-point isolation."""
import sys
from pathlib import Path
from copy import deepcopy
import math
import numpy as np
import pytest
from scipy.integrate import solve_ivp,quad
from scipy.linalg import expm

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import extend_blueprint_model_controls as r


def matrix(p):
    cl,vc,q1,v1,q2,v2=p
    return np.array([[-(cl+q1+q2)/vc,q1/v1,q2/v2],
                     [q1/vc,-q1/v1,0.],[q2/vc,0.,-q2/v2]])


@pytest.mark.parametrize('duration',[0.,.001,.5,2.])
@pytest.mark.parametrize('p',[[3.,10.,8.,30.,2.,100.],[.1,10.,20.,50.,1.,300.],[50.,5.,20.,8.,5.,30.]])
def test_exact_three_compartment_against_independent_matrix_exponential(p,duration):
    dose,start=17.,.3
    times=np.array([0.,start,start+.0001,start+.001,start+.5,start+2.,start+8.,start+40.])
    system=matrix(p)
    expected=[]
    for t in times:
        elapsed=t-start
        if elapsed<0:
            expected.append(0.);continue
        if duration==0:
            state=expm(system*elapsed)@np.array([dose,0.,0.])
        else:
            augmented=np.zeros((4,4));augmented[:3,:3]=system;augmented[0,3]=dose/duration
            state=(expm(augmented*min(elapsed,duration))@np.array([0.,0.,0.,1.]))[:3]
            state=expm(system*max(elapsed-duration,0.))@state
        expected.append(state[0]/p[1])
    actual=r.three_compartment_concentration(p,times,dose_mg=dose,infusion_h=duration,start_h=start)
    np.testing.assert_allclose(actual,expected,rtol=2e-9,atol=1e-12)


@pytest.mark.parametrize('duration',[0.,.5,2.])
def test_independent_ode_and_auc_identity(duration):
    p=np.array([3.,10.,8.,30.,2.,100.]);dose=100.
    system=matrix(p)
    times=np.array([.01,.2,.5,1.,2.,5.,12.,30.])
    if duration:
        during=solve_ivp(lambda t,y:system@y+np.array([dose/duration,0.,0.]),
                          (0,duration),np.zeros(3),rtol=1e-11,atol=1e-12,dense_output=True)
        after=solve_ivp(lambda t,y:system@y,(duration,max(times)),during.y[:,-1],
                        rtol=1e-11,atol=1e-12,dense_output=True)
        expected=np.array([during.sol(t)[0] if t<=duration else after.sol(t)[0] for t in times])/p[1]
    else:
        ode=solve_ivp(lambda t,y:system@y,(0,max(times)),[dose,0.,0.],
                      rtol=1e-11,atol=1e-12,dense_output=True)
        expected=ode.sol(times)[0]/p[1]
    actual=r.three_compartment_concentration(p,times,dose_mg=dose,infusion_h=duration)
    np.testing.assert_allclose(actual,expected,rtol=2e-8,atol=1e-10)
    f=lambda t:r.three_compartment_concentration(p,[t],dose_mg=dose,infusion_h=duration)[0]
    area=quad(f,0,duration,epsabs=1e-9)[0] if duration else 0.
    area+=quad(f,duration,np.inf,epsabs=1e-9)[0]
    assert area==pytest.approx(dose/p[0],rel=1e-9)


@pytest.mark.parametrize('duration',[0.,.001,.5,2.])
def test_zero_second_distribution_degenerates_exactly_to_two_compartments(duration):
    p=[3.,10.,8.,30.];times=np.r_[0.,np.geomspace(.0001,100.,100)]
    expected=r.old.concentration(p,times,dose_mg=100.,infusion_h=duration)
    actual=r.three_compartment_concentration(p+[0.,100.],times,dose_mg=100.,infusion_h=duration)
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
    if duration==0:
        assert actual[0]==pytest.approx(10.)
    else:
        assert actual[0]==0.


def training():
    t=np.array([.25,.5,1.,4.,12.]);p=[3.,10.,8.,30.]
    c=r.old.concentration(p,t,dose_mg=100.,infusion_h=.5)
    return dict(role='calibration_data_reconstruction',time_unit='min',unit='umol/l',molecular_weight_g_mol=200.,
        human=dict(weight_kg=70.),dosing=dict(route='intravenous',dose_mg=100.,infusion_minutes=30.,start_h=0.),
        observations=[dict(time_min=float(tt*60),concentration_umol_per_l=float(cc*1000/200)) for tt,cc in zip(t,c)])


@pytest.mark.parametrize('method',r.METHODS)
def test_fits_only_visible_five_points_without_disk_or_target(method,monkeypatch):
    data=training()
    if method=='one_compartment_cl_prior':
        t=np.array([row['time_min']/60 for row in data['observations']])
        values=r.old.concentration([3.,25.],t,dose_mg=100.,infusion_h=.5)
        for row,value in zip(data['observations'],values):
            row['concentration_umol_per_l']=float(value*1000/200)
    monkeypatch.setattr(r,'read',lambda *args:pytest.fail('fit must not read target/full development'))
    monkeypatch.setattr(r.old,'read',lambda *args:pytest.fail('fit must not read target/full development'))
    fit=r.fit_visible(data,method,'synthetic',prior=3.)
    assert fit['status']=='fitted'
    assert fit['initialization']['observations']==5
    assert fit['subset_log_mse']<1e-6
    assert len(fit['attempts'])==16
    assert all(row.get('nfev',0)<=400 for row in fit['attempts'])
    assert fit['residual_calls']==sum(row['residual_calls'] for row in fit['attempts'])
    if method=='two_compartment_fixed_cl':
        assert fit['selected']['parameters'][0]==3.
        assert fit['free_parameters']==3


def test_three_compartment_initialization_cannot_read_hidden_values():
    data=training();other=deepcopy(data);other['hidden_target']=[1e12]*100
    a,_,_,init=r.initial_three(data,'same')
    b,_,_,_=r.initial_three(other,'same')
    np.testing.assert_array_equal(a,b)
    assert not init['target_observations_used'] and not init['old_fit_used']


def test_rejects_nonfive_observations():
    data=training();data['observations']=data['observations'][:4]
    with pytest.raises(ValueError,match='five'):
        r.fit_visible(data,'three_compartment','x',None)


@pytest.mark.parametrize('bad',[None,0.,-1.,float('nan')])
def test_fixed_clearance_requires_valid_prior(bad):
    with pytest.raises(ValueError,match='Fixed CL'):
        r.fit_visible(training(),'two_compartment_fixed_cl','x',bad)


def test_weight_transfer_for_all_three_compartment_clearances():
    p=np.array([3.,10.,8.,30.,2.,100.])
    np.testing.assert_allclose(r.transferred(p,50,100,'three_compartment','linear_weight'),p*2)
    np.testing.assert_allclose(r.transferred(p,50,100,'three_compartment','allometric_sensitivity'),
                              p*np.array([2**.75,2,2**.75,2,2**.75,2]))


@pytest.mark.parametrize('status,completed',[('fitting',{}),('predictions_locked',{})])
def test_report_refuses_target_read_before_complete_global_lock(monkeypatch,status,completed):
    monkeypatch.setattr(r,'read',lambda p:dict(status=status,protocol={'path':'p','sha256':'x'},completed=completed))
    monkeypatch.setattr(r,'checked',lambda p:dict(jobs={'pending':{}},cases={}))
    monkeypatch.setattr(r,'guard',lambda p:pytest.fail('must refuse before source/scoring access'))
    with pytest.raises(ValueError,match='90 predictions'):
        r.report()


def test_project_output_boundary():
    with pytest.raises(ValueError):
        r.save('/tmp/not_in_project.json',{})
