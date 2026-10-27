"""Acquisition isolation and bookkeeping, using synthetic data only."""
from copy import deepcopy
import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


PROJECT=Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(PROJECT/'scripts'))
    spec=importlib.util.spec_from_file_location('acquisition_runner_test',PROJECT/'scripts/run_endpoint_evidence_design.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'RUN',tmp_path/'run');r.RUN.mkdir()
    monkeypatch.setattr(r.base,'normalized',deepcopy)
    monkeypatch.setattr(r.base,'pins_in',lambda obj,pins:None)
    monkeypatch.setattr(r,'verify',lambda pins:None)
    monkeypatch.setattr(r,'ref',lambda path:{'path':str(Path(path).relative_to(tmp_path)),'sha256':r.base.sha256(path)})
    def checked(source):
        if 'path' not in source:return deepcopy(source)
        path=tmp_path/source['path']
        if r.base.sha256(path)!=source['sha256']:raise ValueError('Changed source hash')
        return r.read(path)
    monkeypatch.setattr(r,'checked',checked)
    return r


def setup(r,monkeypatch,*,feasible=True):
    req=dict(compound_id='synthetic',human={'weight_kg':70.},dosing={'route':'intravenous','dose_mg':5.,'infusion_minutes':1.},
        evidence=[dict(parameter='molecular_weight',unit='g/mol',value=500.)])
    full=dict(compound_id='synthetic',study_id='development',role='calibration_data_reconstruction',matrix='plasma',
        time_unit='min',unit='umol/l',human=req['human'],dosing=req['dosing'],molecular_weight_g_mol=500.,
        observations=[dict(time_min=float(i),concentration_umol_per_l=10./i) for i in range(1,8)])
    five=deepcopy(full);five['observations']=five['observations'][:5]
    split=r.split_development(full,five,req);r.save(r.RUN/'visible.json',split)
    case=dict(case_id='synthetic',drug='Synthetic',cluster_id='s1',requests={'evidence':{'development':req,'target':req}},
        sampling_sources={'full':{'synthetic_full':True}},visible_design=r.ref(r.RUN/'visible.json'))
    jobs={f'synthetic__point_{i:02d}':dict(case_id='synthetic',candidate_id=i,time_h=t)
        for i,t in zip(split['design_only']['candidate_ids'],split['design_only']['candidate_times_h'])}
    p=dict(source_pins={},cases=[case],jobs=jobs,fit_budget_per_formula=16384,starts=16,
        physiology_cache={f'synthetic__development__{f}':{'synthetic_phys':f} for f in r.base.FORMULAS})
    r.save(r.RUN/'protocol.json',p)
    decision=dict(case_id='synthetic',protocol_sha256=r.base.sha256(r.RUN/'protocol.json'),visible_source=case['visible_design'],
        candidate_ids=split['design_only']['candidate_ids'],candidate_times_h=split['design_only']['candidate_times_h'],
        goal_choices={g:0 for g in r.GOALS},generic_choices={'maximum_variance':1,'time_farthest':0},
        details={g:{'status':'selected'} for g in r.GOALS},target_values_used=False,hidden_development_values_used=False,
        resources={'new_profile_calls':0,'committee_simulations':0})
    r.save(r.RUN/'decision.json',decision)
    decisions={'synthetic':{'result':r.ref(r.RUN/'decision.json')}}
    r.save(r.RUN/'decisions_locked.json',dict(protocol=r.ref(r.RUN/'protocol.json'),decisions=decisions,locked_at=1.))
    lock=r.ref(r.RUN/'decisions_locked.json')
    monkeypatch.setattr(r.base,'_training_document',lambda *args:deepcopy(full))
    monkeypatch.setattr(r.prior,'get_phys',lambda *args:{})
    fits=[]
    def fit(predict,request,training,phys,**kwargs):
        fits.append(deepcopy(training));assert len(training['observations'])==6
        assert 'raw_observations' not in training
        return dict(best={'objective':float(len(fits)),'log_parameters':[0.]*4} if feasible else None,
            calls=3,failures=0,variant='blood_partition',call_limit=16384,starts=[{}]*16,
            visible_points=6,no_warm_start=True,target_observations_used=False)
    monkeypatch.setattr(r.old.optimizer,'fit',fit)
    monkeypatch.setattr(r.base.engine,'prediction_grid',lambda req:np.array([.1,1.]))
    monkeypatch.setattr(r.old,'simulate',lambda *args:dict(concentration_mg_l=np.ones(2),mass_balance={}))
    return p,case,jobs,lock,decisions,fits


def test_reveal_is_one_point_after_all_actions_lock_and_selection_precedes_prediction(runner,monkeypatch):
    r=runner;p,case,jobs,lock,_,fits=setup(r,monkeypatch);key=next(iter(jobs));job=jobs[key]
    def simulate(*args):
        assert (r.RUN/'work'/key/'selection.json').exists()
        assert r.checked(lock)['decisions']['synthetic']
        return dict(concentration_mg_l=np.ones(2),mass_balance={})
    monkeypatch.setattr(r.old,'simulate',simulate)
    entry=r.execute(p,key,job,lock);unit=r.checked(entry['result'])
    assert unit['status']=='predicted' and len(fits)==2
    assert all([x['time_min'] for x in fit['observations']]==[1.,2.,3.,4.,5.,6.] for fit in fits)
    assert unit['resources']['objective_calls']==6


def test_incomplete_global_action_lock_blocks_reading_hidden_data(runner,monkeypatch):
    r=runner;p,_,jobs,lock,_,fits=setup(r,monkeypatch)
    path=r.ROOT/lock['path'];receipt=r.read(path);receipt['decisions']={};r.save(path,receipt)
    monkeypatch.setattr(r.base,'_training_document',lambda *a:pytest.fail('Read hidden data before lock'))
    key=next(iter(jobs))
    with pytest.raises(ValueError,match='All policies'):r.execute(p,key,jobs[key],r.ref(path))
    assert fits==[]


def test_candidate_id_time_must_match_frozen_menu(runner,monkeypatch):
    r=runner;p,_,jobs,lock,_,fits=setup(r,monkeypatch);key=next(iter(jobs))
    jobs[key]['time_h']=99.
    with pytest.raises(ValueError,match='absent.*menu'):r.execute(p,key,jobs[key],lock)
    assert fits==[]


def test_infeasible_fits_remain_failed_unit(runner,monkeypatch):
    r=runner;p,_,jobs,lock,_,fits=setup(r,monkeypatch,feasible=False);key=next(iter(jobs))
    entry=r.execute(p,key,jobs[key],lock);unit=r.checked(entry['result'])
    assert unit['status']=='failed' and len(unit['fits'])==2 and len(fits)==2
    assert unit['resources']['prediction_calls']==0


@pytest.mark.parametrize('field,value',[('candidate_id',99),('time_h',99.)])
def test_resume_rejects_different_acquisition(runner,monkeypatch,field,value):
    r=runner;p,_,jobs,lock,_,fits=setup(r,monkeypatch);key=next(iter(jobs))
    r.execute(p,key,jobs[key],lock);changed=deepcopy(jobs[key]);changed[field]=value
    with pytest.raises(ValueError,match='frozen job'):r.execute(p,key,changed,lock)
    assert len(fits)==2


def test_finished_resume_reuses_without_fit(runner,monkeypatch):
    r=runner;p,_,jobs,lock,_,fits=setup(r,monkeypatch);key=next(iter(jobs))
    first=r.execute(p,key,jobs[key],lock);second=r.execute(p,key,jobs[key],lock)
    assert first==second and len(fits)==2


@pytest.mark.parametrize('missing',('endpoint','curve','failure'))
def test_missing_results_cannot_pass_expansion_gate(runner,missing):
    r=runner
    primary={m:{'endpoint':{'n_cases':10},'curve':{'n_cases':10,'estimate':1.}} for m in r.POLICIES}
    pairs={m:{'estimate':.98,'ratio_below_one':6} for m in r.POLICIES};units={'x':{'status':'predicted'}}
    assert all(r.expansion_checks(primary,pairs,units).values())
    if missing=='endpoint':primary['request_directed']['endpoint']['n_cases']=9;pairs['time_farthest']['estimate']=None
    elif missing=='curve':primary['request_directed']['curve']['estimate']=None
    else:units['x']['status']='failed'
    assert not all(r.expansion_checks(primary,pairs,units).values())


def test_report_retains_failed_acquisitions_all_eleven_categories_and_missing_policy(runner,monkeypatch):
    r=runner;p,_,jobs,lock,decisions,_=setup(r,monkeypatch,feasible=False)
    import report_blueprint_validation as report
    units={k:r.execute(p,k,j,lock) for k,j in jobs.items()}
    p.update(baselines={'synthetic':{'status':'failed','reason':'synthetic unavailable'}},limitations=[])
    # Report uses a frozen in-memory test protocol; preserve already locked hash.
    original_checked=r.checked
    def checked(source):
        if source==r.ref(r.RUN/'protocol.json'):return deepcopy(p)
        return original_checked(source)
    monkeypatch.setattr(r,'checked',checked)
    endpoints={e:{'value':1.,'unit':u} for e,u in report.UNITS.items()}
    monkeypatch.setattr(report,'_case_reference',lambda *a:({}, {'unit':'mg/L'},endpoints))
    monkeypatch.setattr(report,'_score_prediction',lambda *a:pytest.fail('Failed unit was scored'))
    state=dict(status='predictions_locked',protocol=r.ref(r.RUN/'protocol.json'),decisions=decisions,
        decisions_lock=lock,units=units,locked_at=2.,started_at=1.)
    r.save(r.RUN/'state.json',state)
    result=r.report();assert result['expand_to_twenty'] is False
    with (r.RUN/'endpoints.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==33 and {x['status'] for x in rows}=={'failed'}
    summary=r.read(r.RUN/'results.json');assert len(summary['records'])==3
    assert all(v['missing_endpoint']==1 for v in summary['summary']['cmax'].values())
    assert r.SCORING['terminal_half_life']=='terminal_half_life'
