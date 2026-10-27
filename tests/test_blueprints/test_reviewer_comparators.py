"""Numerical and visible-data contracts for the new equal-information controls."""
import sys
from pathlib import Path
from copy import deepcopy
import math
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.integrate import quad
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import blueprint_reviewer_comparators as r


def training(model='two'):
    times=np.array([1.,3.,8.,16.])
    pars=[3.,10.,8.,30.] if model=='two' else [3.,25.]
    values=r.concentration(pars,times,dose_mg=100.,infusion_h=.5)
    return dict(role='calibration_data_reconstruction',time_unit='min',unit='umol/l',molecular_weight_g_mol=200.,
        human=dict(weight_kg=70.),dosing=dict(route='intravenous',dose_mg=100.,infusion_minutes=30.,start_h=0.),
        observations=[dict(time_min=float(t*60),concentration_umol_per_l=float(c*1000/200)) for t,c in zip(times,values)])


@pytest.mark.parametrize('duration',[0.,.001,.5,2.])
@pytest.mark.parametrize('parameters',[[3.,25.],[3.,10.,8.,30.]])
def test_matrix_solution_and_auc(parameters,duration):
    p=np.array(parameters)
    matrix=np.array([[-p[0]/p[1]]]) if len(p)==2 else np.array([[-(p[0]+p[2])/p[1],p[2]/p[3]],[p[2]/p[1],-p[2]/p[3]]])
    times=np.array([0.,.3,.30001,.7,2.3,8.,40.]);dose,start=17.,.3
    expected=[]
    for t in times:
        elapsed=t-start
        if elapsed<0:expected.append(0.);continue
        if duration==0:
            state=np.r_[dose,np.zeros(len(matrix)-1)]
            state=expm(matrix*elapsed)@state
        else:
            augmented=np.zeros((len(matrix)+1,len(matrix)+1));augmented[:-1,:-1]=matrix;augmented[0,-1]=dose/duration
            state=(expm(augmented*min(elapsed,duration))@np.r_[np.zeros(len(matrix)),1.])[:-1]
            state=expm(matrix*max(0.,elapsed-duration))@state
        expected.append(state[0]/p[1])
    actual=r.concentration(p,times,dose_mg=dose,infusion_h=duration,start_h=start)
    np.testing.assert_allclose(actual,expected,rtol=3e-10,atol=1e-12)
    # Integrate with the exact infusion boundary explicitly split.
    f=lambda t:float(r.concentration(p,[t],dose_mg=dose,infusion_h=duration)[0])
    area=quad(f,0,duration,epsabs=1e-10)[0] if duration else 0.
    area+=quad(f,duration,np.inf,epsabs=1e-8)[0]
    assert area==pytest.approx(dose/p[0],rel=1e-8)


def test_bolus_is_not_fictitious_infusion():
    p=[3.,10.,8.,30.]
    assert r.concentration(p,[0.],dose_mg=100.,infusion_h=0.)[0]==pytest.approx(10.)
    assert r.concentration(p,[0.],dose_mg=100.,infusion_h=.001)[0]==0.


def test_weight_transfer():
    p=np.array([3.,10.,8.,30.])
    np.testing.assert_allclose(r.transfer_parameters(p,50.,100.),p*2)
    np.testing.assert_allclose(r.transfer_parameters(p,50.,100.,'allometric_sensitivity'),p*np.array([2**.75,2,2**.75,2]))
    with pytest.raises(ValueError):r.transfer_parameters(p,0.,100.)


def test_visible_four_only_no_disk_reads(monkeypatch):
    tr=training('one')
    monkeypatch.setattr(r,'read',lambda *a,**k:pytest.fail('fit must not decode full curve or target'))
    fit=r.fit_visible(tr,'one_compartment','synthetic')
    assert fit['status']=='fitted'
    assert fit['initialization']['observations']==4
    assert fit['subset_log_mse']<1e-12
    np.testing.assert_allclose(fit['selected']['parameters'],[3.,25.],rtol=1e-5)
    assert len(fit['attempts'])==16
    assert all(a.get('nfev',0)<=400 for a in fit['attempts'])


def test_hidden_full_curve_not_accepted():
    tr=training()
    tr['observations']*=2
    with pytest.raises(ValueError,match='four/five'):r.training_arrays(tr)


def test_prior_unavailable_and_forbidden():
    req=dict(compound_id='x',human=dict(weight_kg=70),evidence=[])
    assert not r.clearance_prior(req,set())['available']
    req['evidence']=[dict(parameter='total_clearance',compound_id='x',value=1.,unit='ml/min/kg',
        context=dict(species='human',matrix='plasma',route='intravenous'),source=dict(path='target.json'))]
    assert not r.clearance_prior(req,{'target.json'})['available']


def test_prior_conversion_and_penalty(monkeypatch):
    monkeypatch.setattr(r,'checked',lambda x:{})
    req=dict(compound_id='x',human=dict(weight_kg=50),evidence=[dict(parameter='total_clearance',compound_id='x',value=2.,unit='ml/min/kg',
        context=dict(species='human',matrix='plasma',route='intravenous'),source=dict(path='legal.json',sha256='a'*64))])
    prior=r.clearance_prior(req,set())
    assert prior['value_L_h']==6.
    assert prior['weight']==.1


def test_bootstrap_and_deletion_pairs():
    cases={'a':{'cluster_id':'same'},'b':{'cluster_id':'same'},'c':{'cluster_id':'other'}}
    result=r.pair_stats({'a':2.,'b':4.,'c':8.},{'a':4.,'b':4.,'c':4.},cases)
    assert result['estimate']==pytest.approx(1.)
    assert result['n_cases']==3 and result['n_clusters']==2
    assert (result['improved'],result['worse'],result['tied'])==(1,1,1)
    assert result['ci95'][0]<1<result['ci95'][1]


def test_project_boundary():
    with pytest.raises(ValueError):r.inside('/tmp/non_project_target.json')


def test_report_refuses_prelock(monkeypatch):
    monkeypatch.setattr(r,'read',lambda p:dict(status='fitting',protocol={'path':'p','sha256':'x'},completed={}))
    monkeypatch.setattr(r,'checked',lambda p:dict(jobs={'not_complete':{}}))
    monkeypatch.setattr(r,'guard',lambda p:None)
    with pytest.raises(ValueError,match='Every prediction/failure must lock'):
        r.report()


def test_same_subset_start_independent_of_point_labels():
    tr=training();same=deepcopy(tr);same['unread_hidden_observations']=[1e20]*30
    x,lo,hi,init=r.initial_parameters(tr,'two_compartment','same_case')
    y,_,_,_=r.initial_parameters(same,'two_compartment','same_case')
    np.testing.assert_array_equal(x,y)
    assert init['old_fit_used'] is False and init['target_observations_used'] is False


def test_reference_missing_keeps_null():
    from blueprint_sampling_expansion_data import observed_curve_nca
    out=observed_curve_nca(time_h=[1.,2.,4.,8.],concentration_ng_ml=[8.,4.,1.,.0625],
        dose_ng=1e6,infusion_h=.5,weight_kg=70.)
    assert out['endpoints']['terminal_half_life']['value']==pytest.approx(1.)
    assert out['endpoints']['clearance']['value'] is None
    assert out['endpoints']['auc_inf']['value'] is None


def test_source_metadata_cluster_renaming_keeps_bootstrap_draws(monkeypatch):
    import report_blueprint_reviewer_comparators as report
    pairs=[]
    for cid,new,keep in [('a','doi:z',True),('b','doi:z',True),('c','doi:a',False)]:
        cats={k:False for k in report.CATEGORY_LABELS};cats['patent_source']=not keep
        pairs.append(dict(case_id=cid,drug=cid,target_source_cluster=new,sensitivity_categories=cats,metadata_quality_subset_included=keep))
    p={'cases':{k:dict(drug=k,cluster_id='old1' if k in 'ab' else 'old2',cohort='added_twenty') for k in 'abc'}}
    rows=[dict(case_id=k,info=info,method=method,endpoint=e,transfer='linear_weight',error=str(v))
        for k,v in [('a',2),('b',3),('c',4)] for info,method in [('four','pbpk'),('time_five','pbpk'),('learned_five','pbpk'),('time_five','two_compartment')]
        for e in ('cmax','curve')]
    monkeypatch.setattr(r,'ref',lambda p:{'path':'source','sha256':'f'*64})
    monkeypatch.setattr(r,'checked',lambda p:{'pairs':pairs})
    monkeypatch.setattr(r,'save',lambda *a:None)
    monkeypatch.setattr(r,'csv_save',lambda *a:None)
    result=report.source_sensitivity(p,rows)
    assert result['canonical_group_membership_equals_frozen']
    assert result['bootstrap_draw_labels']=={'a':'old1','b':'old1','c':'old2'}
    assert result['groups']['all_thirty']['strict_metadata_subset']['included']==['a','b']
    assert result['groups']['all_thirty']['strict_metadata_subset']['n_clusters']==1
    for row in rows:row['error']='1000' if row['case_id']=='a' else '1.01'
    changed=report.source_sensitivity(p,rows)
    assert changed['groups']['all_thirty']['strict_metadata_subset']['included']==['a','b']
