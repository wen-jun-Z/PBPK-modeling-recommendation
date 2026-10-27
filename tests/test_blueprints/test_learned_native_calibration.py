"""Synthetic native-adapter contracts; no clinical model or native engine runs."""
from copy import deepcopy
import json
import math

import pytest

from pbpk_kb.blueprint_learning import native_calibration as cal
from test_pk_anchored_blueprint import evidence_case
from test_evidence_preserving_calibration import case


@pytest.fixture
def setup(case, monkeypatch):
    root, request, source, observed = case
    # Native implementation pins are exercised separately on the real project;
    # this fixture substitutes one explicitly frozen synthetic source.
    pins = {"inputs/source.json":cal.sha256(root/"inputs/source.json")}
    monkeypatch.setattr(cal,"_code_pins",lambda _:deepcopy(pins))
    state={"calls":[],"outcomes":[],"pins":pins}
    prior=next(e["value"] for e in request["evidence"] if e["parameter"]=="logp")
    original=cal.assemble_evidence_blueprint(request,root)["snapshot"]["Compounds"][0]
    original_renal=next(p for p in original["Processes"] if p["InternalName"]=="KidneyClearance")

    def native(snapshot, req, output, project_root, *, audit, require_vss=False):
        compound=snapshot["Compounds"][0]
        lip=compound["Lipophilicity"][0]["Parameters"][0]["Value"]
        renal=next(p for p in compound["Processes"] if p["InternalName"]=="KidneyClearance")
        assert next(p for p in renal["Parameters"] if p["Name"]=="Plasma clearance") == next(
            p for p in original_renal["Parameters"] if p["Name"]=="Plasma clearance")
        state["calls"].append({"path":output,"lip":lip,"audit":audit})
        outcome=state["outcomes"].pop(0) if state["outcomes"] else None
        if isinstance(outcome,BaseException): raise outcome
        if outcome is not None: return outcome
        times=[r["time_min"] for r in observed["observations"]]
        values=[r["concentration_umol_per_l"]*math.exp(lip-prior-.625) for r in observed["observations"]]
        curve={"time_min":times,"concentration":values,"unit":"umol/l","time_unit":"min"}
        cal.save(output/"prediction_locked.json",curve)
        cal.save(output/"endpoints.json",{"status":"synthetic_fixture","endpoints":{}})
        return {"status":"predicted","prediction_sha256":cal.sha256(output/"prediction_locked.json")}

    monkeypatch.setattr(cal,"_run_development",native)
    monkeypatch.setattr(cal,"_run",native)
    return root,request,source,observed,state


def fit(setup, **kwargs):
    root,request,source,_,_=setup
    return cal.fit_mapping(reference_request=request,training_source=source,
        output_dir=root/"fit",project_root=root,**{"max_evaluations":12,**kwargs})


def test_actual_search_improves_baseline_and_all_proposals_cost_budget(setup):
    result=fit(setup,initial_points=[{"lipophilicity":-.05,"clearance_scale":1.}])
    assert len(setup[-1]["calls"]) == result["actual_new_calls"] == result["total_native_calls"] == 12
    assert [r["purpose"] for r in result["evaluations"][:2]] == ["evidence_prior","learned_proposal_1"]
    assert result["decision"]["objective"] < result["evaluations"][0]["objective"]
    assert any(r["purpose"]=="pattern_poll" for r in result["evaluations"])
    assert result["status"]=="budget_exhausted" and result["converged"] is False
    assert all(c["audit"] is False for c in setup[-1]["calls"])
    assert all(r["curve_metrics"]["n"]==5 for r in result["evaluations"])
    flags=cal._read(setup[-1]["calls"][0]["path"]/"endpoints.json")
    assert flags["comparison_observations_used"] is False


def test_proposal_bounds_are_explicit_and_invalid_points_rejected(setup):
    result=fit(setup,max_evaluations=2,initial_points=[{"lipophilicity":99.,"clearance_scale":100.}])
    proposal=result["initial_points"][0]
    assert proposal["projected_to_bounds"] is True
    assert proposal["unprojected"]["lipophilicity"]==99.
    assert result["evaluations"][1]["lipophilicity"]==2.4
    assert result["evaluations"][1]["clearance_scale"]==4.
    for points in ([(-.1,1.)]*3,[(float("nan"),1.)],[(0.,0.)]):
        with pytest.raises(ValueError): cal._initial_points(points,-.1)


def test_exact_float_cache_does_not_merge_close_proposals(setup):
    p=math.nextafter(-.1,math.inf)
    result=fit(setup,max_evaluations=3,initial_points=[(-.1,1.),(p,1.)])
    assert result["evaluations"][0]["cache_key"] != result["evaluations"][1]["cache_key"]
    assert result["evaluations"][1]["lipophilicity"]==p
    assert result["cache_hits"]>=1 and len(setup[-1]["calls"])==3


def test_scientific_failure_counts_without_invented_objective(setup):
    setup[-1]["outcomes"]=[{"status":"parameter_readback_failed","reason":["capacity"]}]
    result=fit(setup,max_evaluations=3)
    assert result["failed_evaluation_count"]==1 and result["total_native_calls"]==3
    assert "objective" not in result["evaluations"][0]
    assert result["evaluations"][0]["eligible_for_selection"] is False
    assert result["decision"]["evaluation"]!=1


def test_infrastructure_pause_and_retry_share_budget_without_fake_loss(setup):
    setup[-1]["outcomes"]=[TimeoutError("synthetic native timeout")]
    with pytest.raises(cal.InfrastructurePause): fit(setup,max_evaluations=3)
    assert len(setup[-1]["calls"])==1
    assert not (setup[0]/"fit/mapping_locked.json").exists()
    result=fit(setup,max_evaluations=3)
    assert len(setup[-1]["calls"])==3 and result["total_native_calls"]==3
    assert result["evaluation_count"]==2 and result["infra_retry_calls"]==1
    assert len(result["infrastructure_attempts"])==1
    assert "timeout" in result["infrastructure_attempts"][0]["diagnostic"]["reason"]
    assert setup[-1]["calls"][0]["path"].name=="execution_0"
    assert setup[-1]["calls"][1]["path"].name=="execution_1"


def test_infrastructure_cannot_be_hidden_by_exhausted_budget(setup):
    setup[-1]["outcomes"]=[TimeoutError("synthetic native timeout")]
    with pytest.raises(cal.InfrastructurePause): fit(setup,max_evaluations=1)
    with pytest.raises(cal.InfrastructurePause,match="exhausted its budget"): fit(setup,max_evaluations=1)
    assert len(setup[-1]["calls"])==1 and not (setup[0]/"fit/mapping_locked.json").exists()


def test_resume_reuses_scientific_trials_and_rejects_changed_budget(setup):
    def halt():
        if len(setup[-1]["calls"])==2: raise RuntimeError("project pause")
    with pytest.raises(cal.InfrastructurePause,match="project pause"): fit(setup,before_trial=halt)
    with pytest.raises(ValueError,match="Resume input"): fit(setup,max_evaluations=13)
    result=fit(setup)
    assert result["actual_new_calls"]==10 and result["reused_calls"]==2
    assert len(setup[-1]["calls"])==12
    mapping=setup[0]/result["mapping_path"];before=cal.sha256(mapping)
    again=fit(setup)
    assert again["actual_new_calls"]==0 and again["completed_fit_reused"] is True
    assert again["reused_calls"]==12 and cal.sha256(mapping)==before


def test_resume_rejects_changed_native_artifact(setup):
    def halt():
        if len(setup[-1]["calls"])==1: raise RuntimeError("pause")
    with pytest.raises(cal.InfrastructurePause): fit(setup,before_trial=halt)
    path=setup[-1]["calls"][0]["path"]/"prediction_locked.json"
    path.write_text('{}')
    with pytest.raises(cal.InfrastructurePause,match="resource changed"): fit(setup)
    assert len(setup[-1]["calls"])==1


def test_comparison_source_cannot_be_development_or_initialization(setup):
    root,request,source,_,_=setup
    request["forbidden_sources"]=[{k:source[k] for k in ("path","sha256","location")}]
    with pytest.raises(ValueError,match="cannot enter calibration"): fit(setup)
    request["forbidden_sources"]=[]
    target=root/"inputs/heldout.json";cal.save(target,{"clinical_comparison":"not read by this test"})
    target_ref=cal._ref(target,root);target_ref["location"]="held-out outcomes"
    request["forbidden_sources"]=[target_ref]
    with pytest.raises(ValueError,match="forbidden comparison data"):
        fit(setup,initial_points=[{"lipophilicity":0.,"clearance_scale":1.,"source":target_ref}])
    assert setup[-1]["calls"]==[]


def test_cleanup_retains_scores_best_failure_and_can_reuse_locked_fit(setup):
    setup[-1]["outcomes"]=[{"status":"parameter_readback_failed","reason":"synthetic capacity"}]
    result=fit(setup,max_evaluations=5)
    summary=cal.cleanup_unselected(output_dir=setup[0]/"fit",project_root=setup[0])
    assert summary["removed_count"]==3 and summary["compact_scores_preserved"]==5
    assert all((setup[0]/"fit/trials"/r["cache_key"]/"trial.json").is_file() for r in result["evaluations"])
    assert (setup[0]/result["evaluations"][0]["native_output_dir"]).is_dir()
    assert (setup[0]/result["decision"]["native_output_dir"]).is_dir()
    assert fit(setup,max_evaluations=5)["completed_fit_reused"]
    assert cal.cleanup_unselected(output_dir=setup[0]/"fit",project_root=setup[0])["removed_count"]==0


def test_prediction_is_native_audited_and_completed_call_reused(setup):
    result=fit(setup,max_evaluations=3)
    cal.cleanup_unselected(output_dir=setup[0]/"fit",project_root=setup[0])
    target=deepcopy(setup[1]);target["dosing"]["dose_mg"]=20.
    kwargs=dict(request=target,mapping_path=result["mapping_path"],output_dir=setup[0]/"target",project_root=setup[0])
    first=cal.predict_mapping(**kwargs)
    assert first["status"]=="predicted" and setup[-1]["calls"][-1]["audit"] is True
    second=cal.predict_mapping(**kwargs)
    assert second["completed_prediction_reused"] and second["actual_new_calls"]==0
    assert len(setup[-1]["calls"])==4
    target["dosing"]["dose_mg"]=30.
    with pytest.raises(ValueError,match="differs"): cal.predict_mapping(**kwargs)


def test_prediction_rejects_tampering_and_keeps_partial_diagnostics(setup):
    result=fit(setup,max_evaluations=2)
    kwargs=dict(request=setup[1],mapping_path=result["mapping_path"],output_dir=setup[0]/"target",project_root=setup[0])
    setup[-1]["outcomes"]=[TimeoutError("target test timeout")]
    with pytest.raises(cal.InfrastructurePause): cal.predict_mapping(**kwargs)
    assert (setup[0]/"target/infrastructure_failure.json").is_file()
    calls=len(setup[-1]["calls"])
    with pytest.raises(cal.InfrastructurePause,match="Incomplete prior"): cal.predict_mapping(**kwargs)
    assert len(setup[-1]["calls"])==calls
    kwargs["output_dir"]=setup[0]/"target_explicit_retry"
    cal.predict_mapping(**kwargs)
    (kwargs["output_dir"]/"prediction_locked.json").write_text('{}')
    with pytest.raises(ValueError,match="resource changed"): cal.predict_mapping(**kwargs)


def test_mapping_cannot_substitute_unevaluated_parameters(setup):
    result=fit(setup,max_evaluations=2)
    path=setup[0]/result["mapping_path"]
    mapping=cal._read(path);mapping["decision"]["lipophilicity"]+=.01
    cal.save(path,mapping)
    with pytest.raises(ValueError,match="best actual native trial"):
        cal.predict_mapping(request=setup[1],mapping_path=path,output_dir=setup[0]/"target",project_root=setup[0])
    assert len(setup[-1]["calls"])==2 and not (setup[0]/"target").exists()


def test_cleanup_refuses_unlocked_or_unowned_execution(setup):
    (setup[0]/"unfinished").mkdir()
    with pytest.raises(ValueError,match="locked learning fit"):
        cal.cleanup_unselected(output_dir=setup[0]/"unfinished",project_root=setup[0])
    result=fit(setup,max_evaluations=3)
    unselected=next(r for r in result["evaluations"] if r["cache_key"]!=result["decision"]["cache_key"])
    path=setup[0]/unselected["native_output_dir"]
    moved=path.with_name("kept_native")
    path.rename(moved);path.symlink_to(moved,target_is_directory=True)
    with pytest.raises(ValueError,match="Unsafe native cleanup path"):
        cal.cleanup_unselected(output_dir=setup[0]/"fit",project_root=setup[0])
    assert moved.is_dir()


def test_pattern_reports_true_step_convergence():
    calls=[]
    def evaluate(lip,scale,purpose):
        calls.append((lip,scale))
        return {"evaluation":len(calls),"lipophilicity":lip,"clearance_scale":scale,
            "eligible_for_selection":True,"status":"ok","objective":lip**2+math.log(scale)**2}
    result=cal._pattern(evaluate,0.,[])
    assert result["converged"] and result["optimizer"]["final_step"]<cal.MINIMUM_STEP
    assert result["decision"]["lipophilicity"]==0. and result["decision"]["clearance_scale"]==1.
