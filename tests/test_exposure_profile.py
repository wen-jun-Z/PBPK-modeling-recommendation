"""Visible-window exposure roots, fixed renal input and charged solver budgets."""
from copy import deepcopy
import math
import types

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import exposure_profile as profile


def request():
    c={"species":"human","matrix":"plasma","route":"intravenous"}
    return {"compound_id":"test","human":{"weight_kg":50.},
        "dosing":{"route":"intravenous","dose_mg":100.,"infusion_minutes":30.,"start_h":0.},
        "evidence":[{"parameter":"molecular_weight","value":500.},
            {"parameter":"total_clearance","value":4.,"unit":"ml/min/kg","context":c},
            {"parameter":"renal_clearance","value":1.,"unit":"ml/min/kg","context":c}]}


def training(times=(1.,2.,3.,4.,5.),total_cl=8.):
    times=np.asarray(times,dtype=float)
    return {"compound_id":"test","time_unit":"min","unit":"umol/l",
        "observations":[{"time_min":t*60.,"concentration_umol_per_l":float(20.*math.exp(-t)/total_cl)} for t in times]}


def mock_simulator(records):
    def simulate(_phys,**kw):
        records.append(deepcopy(kw))
        t=np.asarray(kw["times_h"])
        shape=kw["log_parameters"]
        # Positive and strictly decreasing in systemic CL for fixed shape.
        p=10.*np.exp(-t/math.exp(shape["bidirectional_exchange_scale"]))/kw["total_clearance_l_h"]
        p*=math.exp(-shape["partition_scale"])
        return {"concentration_mg_l":p}
    return simulate


def once(fun,x0,**kwargs):
    fun(x0)
    return types.SimpleNamespace(success=True,message="single evaluation")


def test_visible_auc_uses_declared_units_and_never_adds_zero_or_tail():
    t,c,w,auc=profile.visible_window(request(),training())
    np.testing.assert_array_equal(t,[1,2,3,4,5])
    np.testing.assert_array_equal(w,[.5,1,1,1,.5])
    assert auc==pytest.approx(np.trapezoid(c,t))
    assert t[0]==1. and len(t)==5


def test_known_monotone_clearance_root_and_unchanged_source_request(monkeypatch):
    calls=[]
    monkeypatch.setattr(profile,"simulate",mock_simulator(calls))
    monkeypatch.setattr(profile,"least_squares",once)
    req=request();original=deepcopy(req)
    result=profile.fit({"circulation":{"cardiac_flow_l_h":100.}},req,training(),budget=100,starts=1)
    assert req==original and result["status"]=="fitted"
    best=result["best"]
    assert best["profiled_total_clearance_l_h"]==pytest.approx(8.,rel=1e-8)
    assert best["profiled_nonrenal_clearance_l_h"]==pytest.approx(5.,rel=1e-8)
    assert best["parameters"]["clearance_scale"]==pytest.approx(5./9.,rel=1e-8)
    assert best["relative_auc_error"]<1e-8 and best["development_gmfe"]==pytest.approx(1.)
    assert result["calls"]==len(calls)
    assert all(c["renal_clearance_l_h"]==3. and c["log_parameters"]["clearance_scale"]==0. for c in calls)
    assert result["root_cache_hits"]>0


def test_repeated_root_queries_are_cached_not_counted_as_new_simulations(monkeypatch):
    calls=[]
    monkeypatch.setattr(profile,"simulate",mock_simulator(calls))
    def repeated(fun,x0,**kwargs):
        a=fun(x0);b=fun(x0);np.testing.assert_array_equal(a,b)
        return types.SimpleNamespace(success=True,message="twice")
    monkeypatch.setattr(profile,"least_squares",repeated)
    out=profile.fit({"circulation":{"cardiac_flow_l_h":100.}},request(),training(),budget=100,starts=1)
    assert out["profile_cache_hits"]==1 and out["calls"]==len(calls)


def test_hard_budget_includes_incomplete_root_and_all_initial_starts(monkeypatch):
    calls=[]
    def shape_independent(_phys,**kw):
        calls.append(kw)
        return {"concentration_mg_l":10.*np.exp(-np.asarray(kw["times_h"]))/kw["total_clearance_l_h"]}
    monkeypatch.setattr(profile,"simulate",shape_independent)
    monkeypatch.setattr(profile,"least_squares",once)
    out=profile.fit({"circulation":{"cardiac_flow_l_h":100.}},request(),training(),budget=9,starts=3)
    assert out["calls"]==len(calls)==9
    assert [s["calls"] for s in out["starts"]]==[3,3,3]
    assert all(s["status"]=="call_budget_exhausted" for s in out["starts"])
    assert out["best"] is None


def test_failed_model_attempts_consume_budget(monkeypatch):
    calls=[]
    def fail(*_,**__):
        calls.append(1);raise ValueError("simulator failure")
    monkeypatch.setattr(profile,"simulate",fail)
    monkeypatch.setattr(profile,"least_squares",once)
    out=profile.fit({"circulation":{"cardiac_flow_l_h":100.}},request(),training(),budget=16,starts=4)
    assert out["calls"]==out["failures"]==len(calls)==4
    assert out["status"]=="failed" and out["infeasible_reasons"]["model_evaluation_failed"]==4


def test_no_bracket_is_not_interpolated_or_clipped_into_a_fake_solution(monkeypatch):
    monkeypatch.setattr(profile,"simulate",lambda _phys,**kw:{"concentration_mg_l":np.full(len(kw["times_h"]),1e-8)})
    monkeypatch.setattr(profile,"least_squares",once)
    out=profile.fit({"circulation":{"cardiac_flow_l_h":100.}},request(),training(),budget=100,starts=1)
    assert out["best"] is None and out["calls"]==2
    assert out["infeasible_reasons"]=={"observed_auc_exceeds_min_clearance_exposure":1}


def test_reverse_monotonicity_is_rejected(monkeypatch):
    monkeypatch.setattr(profile,"simulate",lambda _phys,**kw:{"concentration_mg_l":np.full(len(kw["times_h"]),kw["total_clearance_l_h"])})
    monkeypatch.setattr(profile,"least_squares",once)
    out=profile.fit({"circulation":{"cardiac_flow_l_h":100.}},request(),training(),budget=100,starts=1)
    assert out["infeasible_reasons"]=={"nonmonotone_exposure_bracket":1}


def test_source_guard_failure_is_not_mislabeled_as_a_failed_cl_trial(monkeypatch):
    monkeypatch.setattr(profile,"simulate",lambda *_args,**_kw:pytest.fail("No simulation after source-integrity failure"))
    monkeypatch.setattr(profile,"least_squares",once)
    def stop():raise RuntimeError("source changed")
    with pytest.raises(RuntimeError,match="source changed"):
        profile.fit({"circulation":{"cardiac_flow_l_h":100.}},request(),training(),budget=100,starts=1,before_trial=stop)


@pytest.mark.parametrize("exception_type",[RuntimeError,ValueError])
def test_guard_failure_inside_brent_callback_aborts_the_whole_fit(monkeypatch,exception_type):
    calls=[]
    guard_calls=[]
    monkeypatch.setattr(profile,"simulate",mock_simulator(calls))
    monkeypatch.setattr(profile,"least_squares",once)
    def stop_after_bracket():
        guard_calls.append(1)
        if len(guard_calls)==3:raise exception_type("source changed inside root")
    with pytest.raises(exception_type,match="source changed inside root"):
        profile.fit({"circulation":{"cardiac_flow_l_h":100.}},request(),training(),budget=100,starts=1,before_trial=stop_after_bracket)
    assert len(calls)==2


def test_five_point_fit_uses_only_visible_points_even_if_hidden_data_keys_exist(monkeypatch):
    calls=[]
    monkeypatch.setattr(profile,"simulate",mock_simulator(calls))
    monkeypatch.setattr(profile,"least_squares",once)
    doc=training((.1,.5,1.,2.,8.));doc["full_curve"]=training(np.arange(1.,51.))
    out=profile.fit({"circulation":{"cardiac_flow_l_h":100.}},request(),doc,budget=100,starts=1)
    assert out["visible_training_points"]==5
    for call in calls:np.testing.assert_array_equal(call["times_h"],[.1,.5,1.,2.,8.])
    assert out["added_zero_time_or_extrapolated_tail"] is False


def test_outer_shape_optimization_improves_shape_without_losing_auc_constraint(monkeypatch):
    calls=[]
    monkeypatch.setattr(profile,"simulate",mock_simulator(calls))
    doc=training()
    for r in doc["observations"]:r["concentration_umol_per_l"]=20.*math.exp(-(r["time_min"]/60.)/1.5)/8.
    out=profile.fit({"circulation":{"cardiac_flow_l_h":100.}},request(),doc,budget=400,starts=1)
    assert out["best"] is not None and out["best"]["relative_auc_error"]<1e-8
    assert out["best"]["development_ln_rmse"]<.02
    assert out["calls"]==len(calls)<=400
    assert out["best"]["parameters"]["bidirectional_exchange_scale"]==pytest.approx(1.5,rel=.02)


@pytest.mark.parametrize("mutate",[
    lambda d:d["observations"].__setitem__(1,deepcopy(d["observations"][0])),
    lambda d:d["observations"][0].update(concentration_umol_per_l=0),
    lambda d:d.update(unit="ng/ml"),
])
def test_invalid_visible_data_is_rejected_before_any_model_call(monkeypatch,mutate):
    doc=training();mutate(doc)
    monkeypatch.setattr(profile,"simulate",lambda *_args,**_kw:pytest.fail("Invalid data must not be simulated"))
    with pytest.raises(ValueError):
        profile.fit({"circulation":{"cardiac_flow_l_h":100.}},request(),doc,budget=100,starts=1)


def test_fixed_initial_starts_and_engineering_range_are_explicit(monkeypatch):
    np.testing.assert_array_equal(profile.initial_starts(3),profile.initial_starts(16)[:3])
    monkeypatch.setattr(profile,"simulate",mock_simulator([]))
    monkeypatch.setattr(profile,"least_squares",once)
    out=profile.fit({"circulation":{"cardiac_flow_l_h":200.}},request(),training(),budget=100,starts=1)
    assert out["nonrenal_clearance_engineering_bounds_l_h"]==[1e-4,2000.]
    assert out["engineering_bounds_are_physiological_certification"] is False
    assert out["optimizer"]["outer_dimensions"]==2 and out["optimizer"]["CL_prior_penalty"]==0.
