"""Bounded runner checks with mathematical curves and isolated project fixtures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    spec = importlib.util.spec_from_file_location("condition_matched_runner_test",ROOT/"scripts/run_condition_matched_selection.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training(n=5):
    return {"compound_id":"test","role":"calibration_data_reconstruction","matrix":"plasma",
        "time_unit":"min","unit":"umol/l",
        "observations":[{"time_min":float(i+1),"concentration_umol_per_l":2.*np.exp(-(i+1)/10.)} for i in range(n)]}


def request():
    return {"compound_id":"test","human":{"weight_kg":70.},
        "dosing":{"dose_mg":100.,"infusion_minutes":30.},"evidence":[
            {"parameter":"molecular_weight","value":500.},
            {"parameter":"elimination_scope","value":"hepatic_only"}]}


def hungry_optimizer(fun,x0,**kwargs):
    # This intentionally emulates an optimizer's unreported finite-difference
    # evaluations: only the runner's actual residual-call count may stop it.
    for j in range(10000):
        fun(np.asarray(x0)+np.array([j*1e-10,0.,0.]))
    raise AssertionError("Hard objective-call limit did not interrupt fitting")


def test_allocation_preserves_total_starts_and_calls_for_different_candidate_counts(runner):
    for count in (2,4,6,8,10,12,16):
        calls = runner.allocated(4096,count)
        starts = runner.allocated(16,count)
        assert sum(calls)==4096 and sum(starts)==16
        assert max(calls)-min(calls)<=1 and min(starts)>=1
    with pytest.raises(ValueError):
        runner.allocated(16,17)


def test_objective_limit_counts_finite_differences_failures_and_each_start(runner,monkeypatch):
    calls = []
    def simulate(_phys,_req,times,*_):
        calls.append(times.copy())
        if len(calls)%3==0:
            raise ValueError("Deliberate failed trial")
        return {"concentration_mg_l":np.exp(-np.asarray(times))}
    monkeypatch.setattr(runner.engine,"_simulate",simulate)
    monkeypatch.setattr(runner,"least_squares",hungry_optimizer)
    result = runner.fit_cell({},request(),training(),17,3,lambda:None)
    assert len(calls)==result["calls"]==17
    assert result["failures"]==5
    assert [s["calls"] for s in result["starts"]]==[6,6,5]
    assert all(s["status"]=="call_budget_exhausted" for s in result["starts"])
    assert result["no_warm_start"] is True


def test_malformed_prediction_cannot_broadcast_into_a_false_curve_fit(runner,monkeypatch):
    monkeypatch.setattr(runner.engine,"_simulate",lambda *_:{"concentration_mg_l":np.array([1.])})
    monkeypatch.setattr(runner,"least_squares",hungry_optimizer)
    result = runner.fit_cell({},request(),training(),7,1,lambda:None)
    assert result["calls"]==7 and result["failures"]==7
    assert result["best"] is None


def test_source_guard_failure_is_not_swallowed_as_an_optimizer_trial(runner,monkeypatch):
    monkeypatch.setattr(runner,"least_squares",hungry_optimizer)
    monkeypatch.setattr(runner.engine,"_simulate",lambda *_:pytest.fail("Simulation must not run after source change"))
    def stop():
        raise RuntimeError("Source hash changed")
    with pytest.raises(RuntimeError,match="Source hash changed"):
        runner.fit_cell({},request(),training(),7,1,stop)


def test_guard_stops_on_changed_project_source_and_pause_file(runner,monkeypatch,tmp_path):
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner,"RUN",tmp_path)
    source = tmp_path/"input.json"
    source.write_text("{}")
    guard = runner.Guard({"source_pins":{"input.json":runner.sha256(source)}})
    guard(force=True)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError,match="Source changed"):
        guard(force=True)
    (tmp_path/"PAUSE").write_text("{}")
    with pytest.raises(RuntimeError,match="Paused"):
        guard()


def test_make_request_preserves_selected_net_renal_value_and_weight_conversion(runner,monkeypatch,tmp_path):
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    class Validated:
        def __init__(self,value):self.value=value
        def model_dump(self,**_):return deepcopy(self.value)
    monkeypatch.setattr(runner,"EvidenceBlueprintRequest",types.SimpleNamespace(model_validate=Validated))
    original = request()
    original["evidence"] += [{"parameter":"hepatic_clearance","value":99.},
                              {"parameter":"renal_clearance","value":98.}]
    saved = deepcopy(original)
    bundle = {"records":[{"candidate_id":p,"parameter":p,"value":value,"basis":"measured",
        "source":{"path":"source.pdf","sha256":"a"*64,"location":"Table 1"},
        "context":{"species":"human","matrix":"plasma","route":"intravenous","study_id":"study","arm_id":"IV"}}
        for p,value in (("total_clearance",2.),("renal_clearance",.3),("fraction_unbound",.5))]}
    updated = runner.make_request(original,bundle,tmp_path)
    assert original==saved
    values = {r["parameter"]:r["value"] for r in updated["evidence"]}
    assert values["renal_clearance"]==.3 and values["total_clearance"]==2.
    assert "hepatic_clearance" not in values and values["elimination_scope"]=="hepatic_and_renal"
    cl = runner.engine.clearance_inputs(updated)
    assert cl["renal_input_fixed"] is True
    assert cl["renal_clearance_l_h"]==pytest.approx(.3*70*.06)
    assert cl["total_clearance_l_h"]==pytest.approx(2.*70*.06)
    import pbpk_kb.blueprint_policy.permeability_model as permeability
    seen = []
    def simulate(_phys,**kwargs):
        seen.append(kwargs)
        return {"concentration_mg_l":[1.]}
    monkeypatch.setattr(permeability,"simulate",simulate)
    for scale in (.5,2.):
        runner.engine._simulate({},updated,[1.],"flow_exchange",runner.engine.PARAMETERS["flow_exchange"],np.log([scale,1.,1.]))
    assert seen[0]["renal_clearance_l_h"]==seen[1]["renal_clearance_l_h"]==cl["renal_clearance_l_h"]


def test_execute_reads_only_declared_five_points_and_locks_target_prediction(runner,monkeypatch,tmp_path):
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(runner,"RUN",tmp_path)
    doc = training(5)
    source = tmp_path/"five.json";source.write_text(json.dumps(doc))
    forbidden = tmp_path/"full.json";forbidden.write_text(json.dumps(training(15)))
    original = request()
    original["forbidden_sources"]=[{"path":"full.json","sha256":runner.sha256(forbidden)}]
    source_ref = {"path":"five.json","sha256":runner.sha256(source),"role":"calibration_data_reconstruction"}
    case = {"case_id":"case","requests":{"evidence":{"development":{"path":"dev_request"},"target":{"path":"target_request"}}}}
    checked = []
    def read_request(ref):
        checked.append(ref["path"])
        return {**deepcopy(original),"test_phase":ref["path"]}
    monkeypatch.setattr(runner,"checked",read_request)
    monkeypatch.setattr(runner,"make_request",lambda req,*_:deepcopy(req))
    monkeypatch.setattr(runner.engine,"_physiology",lambda *_:{})
    monkeypatch.setattr(runner.engine,"prediction_grid",lambda _req:np.array([0.,1.,2.]))
    monkeypatch.setattr(runner,"summarize_extended_prediction",lambda *_:{"fixture":True})
    times_seen = []
    def simulate(_phys,req,times,*_):
        if req["test_phase"]=="dev_request":times_seen.append(np.asarray(times).tolist())
        return {"concentration_mg_l":np.exp(-np.asarray(times)),"observation":"fixture","mass_balance":{"ok":True}}
    monkeypatch.setattr(runner.engine,"_simulate",simulate)
    monkeypatch.setattr(runner,"least_squares",hungry_optimizer)
    protocol = {"cases":[case]}
    (tmp_path/"protocol.json").write_text(json.dumps(protocol))
    job = {"case_id":"case","sampling":"five","method":"joint","selection_status":"ready",
        "training_source":source_ref,"configs":[{"bundle":{"id":"bundle","selected":{},"records":[]},
            "formula":"PKSimStandard","call_limit":7,"start_count":1}]}
    result = runner.execute(protocol,job,"case__five__joint",lambda:None)
    assert checked==["dev_request","target_request"]
    assert times_seen and all(t==[r["time_min"]/60 for r in doc["observations"]] for t in times_seen)
    assert result["status"]=="predicted" and result["target_observations_used"] is False
    assert (tmp_path/result["prediction"]["path"]).exists()
    # A changed cache request cannot silently masquerade as a matching resume.
    result_file = tmp_path/"work/case__five__joint/result.json"
    result_file.unlink()
    fit = tmp_path/"work/case__five__joint/candidate_00/fit.json"
    changed = json.loads(fit.read_text());changed["request"]["test_phase"]="altered_request"
    fit.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="(?i)(resume|changed|differ|match|cached)"):
        runner.execute(protocol,job,"case__five__joint",lambda:None)


def _report_import_stub(monkeypatch):
    m = types.ModuleType("report_blueprint_validation")
    def forbidden(*_,**__):pytest.fail("Target observations must not be opened before every prediction attempt is locked")
    for name in ("Store","_case_reference","_score_prediction","paired_ratio"):
        setattr(m,name,forbidden)
    monkeypatch.setitem(sys.modules,"report_blueprint_validation",m)


@pytest.mark.parametrize("state",[
    {"status":"running","all_predictions_locked_before_scoring":False,"units":{}},
    {"status":"predictions_locked","all_predictions_locked_before_scoring":True,"units":{}},
])
def test_scoring_refuses_unlocked_or_incomplete_prediction_set(runner,monkeypatch,tmp_path,state):
    _report_import_stub(monkeypatch)
    monkeypatch.setattr(runner,"RUN",tmp_path)
    protocol = {"jobs":{"expected_job":{}},"source_pins":{},"cases":[]}
    state = {**state,"protocol":{},"output_pins":{}}
    monkeypatch.setattr(runner,"read",lambda p:protocol if Path(p).name=="protocol.json" else state)
    monkeypatch.setattr(runner,"checked",lambda _:protocol)
    with pytest.raises(ValueError):runner.report()


def test_frozen_controls_are_explicitly_not_compute_matched(runner,monkeypatch,tmp_path):
    # Prepare with tiny frozen metadata, no models, real files or clinical data.
    monkeypatch.setattr(runner,"RUN",tmp_path)
    monkeypatch.setattr(runner,"PAPER",tmp_path/"paper")
    case = {"case_id":"case","sampling_sources":{"full":{},"five":{}}}
    candidates = {"inventory":[{"case_id":"case","eligible_for_multisource_experiment":True,"selector_context":{}}],"candidates":[]}
    computations = {f"case__{sampling}__{method}":{"status":"predicted"}
        for sampling in ("full","five") for method in ("flow_exchange","two_compartment")}
    def read(path):
        p=Path(path)
        if p.name=="candidates.json":return candidates
        if p.name=="state.json":return {"computations":computations}
        return {"cases":[case]}
    monkeypatch.setattr(runner,"read",read)
    monkeypatch.setattr(runner,"sha256",lambda _:"a"*64)
    monkeypatch.setattr(runner,"pins_in",lambda *_:None)
    monkeypatch.setattr(runner,"verify",lambda *_:None)
    monkeypatch.setattr(runner.engine,"_code_pins",lambda _: {})
    monkeypatch.setattr(runner,"select_bundles",lambda *_,**__:{"bundles":[{"id":"a"},{"id":"b"}],"status":"ready"})
    result=runner.prepare()
    protocol=json.loads((tmp_path/"protocol.json").read_text())
    assert result["controls_reused"]==4
    assert protocol["original_paper_control_budget_differs"] is True
    for job in protocol["jobs"].values():
        assert sum(c["call_limit"] for c in job["configs"])==4096
        assert sum(c["start_count"] for c in job["configs"])==16
