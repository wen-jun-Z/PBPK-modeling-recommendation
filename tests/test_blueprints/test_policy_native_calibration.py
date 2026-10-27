"""Policy calibration contracts with synthetic native calls, no clinical run."""
from copy import deepcopy
import math

import numpy as np
import pytest

from pbpk_kb.blueprint_policy import native_calibration as cal
from test_pk_anchored_blueprint import evidence_case
from test_evidence_preserving_calibration import case


@pytest.fixture
def setup(case,monkeypatch):
    root,request,source,observed=case
    pins={"inputs/source.json":cal.sha256(root/"inputs/source.json")}
    monkeypatch.setattr(cal,"_code_pins",lambda _:deepcopy(pins))
    prior=next(e["value"] for e in request["evidence"] if e["parameter"]=="logp")
    original=cal.assemble_evidence_blueprint(request,root)["snapshot"]["Compounds"][0]
    original_processes={p["InternalName"]:p for p in original["Processes"]}

    def plasma(process):
        return next(p for p in process["Parameters"] if p["Name"]=="Plasma clearance")

    state={"calls":[],"outcomes":[],"mode":"independent","prior":prior}

    def native(snapshot,req,output,project_root,*,audit,require_vss=False):
        compound=snapshot["Compounds"][0]
        processes={p["InternalName"]:p for p in compound["Processes"]}
        assert plasma(processes["KidneyClearance"])==plasma(original_processes["KidneyClearance"])
        lip=compound["Lipophilicity"][0]["Parameters"][0]["Value"]
        scale=plasma(processes["LiverClearance"])["Value"]/plasma(original_processes["LiverClearance"])["Value"]
        state["calls"].append({"lip":lip,"scale":scale,"path":output,"audit":audit})
        outcome=state["outcomes"].pop(0) if state["outcomes"] else None
        if isinstance(outcome,BaseException):raise outcome
        if outcome is not None:return outcome
        z1=(lip-prior)/2.5;z2=math.log(scale)/math.log(4)
        shape=np.linspace(-1,1,len(observed["observations"]))
        if state["mode"]=="independent":delta=z1+z2*shape-.2
        elif state["mode"]=="lip_only":delta=np.full_like(shape,z1-.2)
        elif state["mode"]=="collinear":delta=np.full_like(shape,z1+2*z2-.2)
        elif state["mode"]=="weak":delta=.001*z1+.002*z2*shape
        else:raise AssertionError(state["mode"])
        values=np.asarray([r["concentration_umol_per_l"] for r in observed["observations"]])*np.exp(delta)
        curve={"time_min":[r["time_min"] for r in observed["observations"]],"concentration":values.tolist(),
               "unit":"umol/l","time_unit":"min"}
        cal.save(output/"prediction_locked.json",curve)
        cal.save(output/"endpoints.json",{"status":"synthetic_fixture","endpoints":{}})
        return {"status":"predicted","prediction_sha256":cal.sha256(output/"prediction_locked.json")}

    monkeypatch.setattr(cal,"_run_development",native)
    monkeypatch.setattr(cal,"_run",native)
    return root,request,source,observed,state


def fit(setup,**kwargs):
    root,request,source,_,_=setup
    return cal.fit_mapping(reference_request=request,training_source=source,
        output_dir=root/"fit",project_root=root,**{"max_evaluations":16,**kwargs})


def test_real_central_differences_and_all_calls_share_budget(setup):
    result=fit(setup,screen_parameters=True)
    assert result["total_native_calls"]==result["actual_new_calls"]==16==len(setup[-1]["calls"])
    screen=result["screening"]
    assert screen["status"]=="screened" and result["active_parameters"]==list(cal.PARAMETERS)
    assert screen["probe_evaluations"]==[2,3,4,5]
    assert np.asarray(screen["jacobian"])[:,0]==pytest.approx(np.ones(5))
    assert np.asarray(screen["jacobian"])[:,1]==pytest.approx(np.linspace(-1,1,5))
    assert screen["rank_analysis"]["rank"]==2 and screen["rank_analysis"]["observation_count"]==5
    assert all(not row["eligible_for_selection"] for row in result["evaluations"][1:5])
    assert not result["decision"]["purpose"].startswith("sensitivity_")
    assert all(row["curve_metrics"]["n"]==5 for row in result["evaluations"])
    assert result["decision"]["objective"]<result["evaluations"][0]["objective"]


def test_single_allowed_parameter_never_changes_protected_axis(setup):
    result=fit(setup,allowed_parameters=("clearance_scale",),screen_parameters=True)
    assert result["screening"]["probe_evaluations"]==[2,3]
    assert all(call["lip"]==setup[-1]["prior"] for call in setup[-1]["calls"])
    assert result["active_parameters"]==["clearance_scale"]
    assert result["optimizer"]["frozen_at_evidence_parameters"]==["lipophilicity"]


def test_zero_response_axis_frozen_after_diagnostics(setup):
    setup[-1]["mode"]="lip_only"
    result=fit(setup,screen_parameters=True)
    assert result["active_parameters"]==["lipophilicity"]
    assert result["screening"]["weak_response_parameters"]==["clearance_scale"]
    assert all(row["clearance_scale"]==1. for row in result["evaluations"] if row["purpose"]=="pattern_poll")
    assert result["decision"]["clearance_scale"]==1.


def test_collinear_axes_keep_stronger_direction_and_fixed_evidence_origin(setup):
    setup[-1]["mode"]="collinear"
    result=fit(setup,screen_parameters=True)
    assert result["screening"]["rank_analysis"]["rank"]==1
    assert result["active_parameters"]==["clearance_scale"]
    assert all(row["lipophilicity"]==setup[-1]["prior"] for row in result["evaluations"] if row["purpose"]=="pattern_poll")


def test_absolute_response_gate_can_retain_only_evidence_point(setup):
    setup[-1]["mode"]="weak"
    result=fit(setup,screen_parameters=True)
    assert result["active_parameters"]==[] and result["status"]=="fixed_evidence_only"
    assert result["total_native_calls"]==5 and result["decision"]["evaluation"]==1
    assert result["converged"] is False
    assert result["screening"]["rank_analysis"]["rank"]==0


def test_empty_allowed_set_uses_one_call_no_spurious_fit(setup):
    result=fit(setup,allowed_parameters=(),screen_parameters=True)
    assert result["total_native_calls"]==1 and result["active_parameters"]==[]
    assert result["screening"]["status"]=="no_allowed_parameters"


def test_failed_scientific_probe_is_not_low_sensitivity_and_preserved(setup):
    setup[-1]["outcomes"]=[None,{"status":"parameter_readback_failed","reason":"synthetic capacity"}]
    result=fit(setup,screen_parameters=True)
    assert result["screening"]["status"]=="screen_failed" and result["screening"]["fallback_used"]
    assert result["active_parameters"]==list(cal.PARAMETERS)
    assert result["failed_evaluation_count"]==1 and result["total_native_calls"]==16
    assert result["screening"]["failed_probes"][0]["evaluation"]==2
    cal.cleanup_unselected(output_dir=setup[0]/"fit",project_root=setup[0])
    assert (setup[0]/result["evaluations"][1]["native_output_dir"]).is_dir()
    assert all((setup[0]/"fit/trials"/row["cache_key"]/"trial.json").is_file() for row in result["evaluations"])
    assert fit(setup,screen_parameters=True)["completed_fit_reused"]


def test_exhausted_diagnostic_budget_falls_back_and_does_not_overrun(setup):
    result=fit(setup,screen_parameters=True,max_evaluations=3)
    assert result["total_native_calls"]==3 and result["screening"]["status"]=="screen_failed"
    assert result["active_parameters"]==list(cal.PARAMETERS)
    assert result["decision"]["evaluation"]==1


def test_infrastructure_failure_pauses_then_one_retry_charged(setup):
    setup[-1]["outcomes"]=[None,TimeoutError("synthetic native timeout")]
    with pytest.raises(cal.InfrastructurePause):fit(setup,screen_parameters=True)
    assert not (setup[0]/"fit/mapping_locked.json").exists()
    result=fit(setup,screen_parameters=True)
    assert result["total_native_calls"]==16 and result["infra_retry_calls"]==1
    assert result["evaluation_count"]==15
    assert len(result["infrastructure_attempts"])==1


def test_pause_replay_identical_screen_and_reject_changed_policy(setup):
    def stop():
        if len(setup[-1]["calls"])==4:raise RuntimeError("project pause")
    with pytest.raises(cal.InfrastructurePause,match="project pause"):
        fit(setup,screen_parameters=True,before_trial=stop)
    with pytest.raises(ValueError,match="Resume input"):
        fit(setup,screen_parameters=False)
    with pytest.raises(ValueError,match="Resume input"):
        fit(setup,screen_parameters=True,allowed_parameters=("lipophilicity",))
    result=fit(setup,screen_parameters=True)
    assert result["actual_new_calls"]==12 and result["reused_calls"]==4
    assert result["screening"]["rank_analysis"]["rank"]==2
    assert result["total_native_calls"]==16


def test_target_prediction_requires_exact_locked_identity_and_can_resume(setup):
    result=fit(setup,screen_parameters=True)
    target=deepcopy(setup[1]);target["dosing"]["dose_mg"]*=2
    kwargs=dict(request=target,mapping_path=result["mapping_path"],output_dir=setup[0]/"target",project_root=setup[0])
    output=cal.predict_mapping(**kwargs)
    assert output["status"]=="predicted" and setup[-1]["calls"][-1]["audit"] is True
    assert cal.predict_mapping(**kwargs)["completed_prediction_reused"]
    target["dosing"]["dose_mg"]*=2
    with pytest.raises(ValueError,match="differs"):cal.predict_mapping(**kwargs)
    mapping=setup[0]/result["mapping_path"]
    changed=cal._read(mapping);changed["allowed_parameters"]=["lipophilicity"];cal.save(mapping,changed)
    with pytest.raises(ValueError,match="policy changed"):cal.predict_mapping(**kwargs)


def test_comparison_source_cannot_enter_development(setup):
    setup[1]["forbidden_sources"]=[{key:setup[2][key] for key in ("path","sha256","location")}]
    with pytest.raises(ValueError,match="cannot enter calibration"):fit(setup)
    assert setup[-1]["calls"]==[]


@pytest.mark.parametrize("parameters",["lipophilicity",("unknown",),("lipophilicity","lipophilicity"),None])
def test_rejects_invalid_parameter_policy(setup,parameters):
    with pytest.raises(ValueError):fit(setup,allowed_parameters=parameters)
    assert setup[-1]["calls"]==[]


def test_cleanup_rejects_alias_of_owned_execution(setup):
    result=fit(setup,screen_parameters=True)
    probe=result["evaluations"][1];path=setup[0]/probe["native_output_dir"]
    moved=path.with_name("retained");path.rename(moved);path.symlink_to(moved,target_is_directory=True)
    with pytest.raises(ValueError,match="Unsafe native cleanup path"):
        cal.cleanup_unselected(output_dir=setup[0]/"fit",project_root=setup[0])
    assert moved.is_dir()
