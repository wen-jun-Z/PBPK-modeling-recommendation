"""Evidence-preserving mapping and restart contracts; no native engine runs."""
from copy import deepcopy
import json
import shutil

import pytest

from pbpk_kb.blueprints import evidence_preserving_calibration as ep
from pbpk_kb.blueprints import evidence_blueprint as eb
from pbpk_kb.blueprints import pk_anchored_blueprint as old
from test_pk_anchored_blueprint import evidence_case
from test_development_curve_blueprint import _native_success


@pytest.fixture
def case(evidence_case):
    root, request, _ = evidence_case
    request = eb.EvidenceBlueprintRequest.model_validate(request).model_dump(mode="json", exclude_none=True)
    doc = {"compound_id": request["compound_id"], "study_id": "test:independent-development",
        "role": "calibration_data_reconstruction", "matrix": "plasma", "time_unit": "min", "unit": "umol/l",
        "human": deepcopy(request["human"]), "dosing": deepcopy(request["dosing"]),
        "observations": [{"time_min": t, "concentration_umol_per_l": v}
                         for t, v in zip([15., 60., 120., 240., 480.], [1., .5, .3, .1, .01])],
        "raw_observations": [{"time_min": 30, "concentration_umol_per_l": 1e8}],
        "sampling": {"selected_count": 5, "original_count": 6}}
    path = root / "inputs/development_five.json"
    path.write_text(json.dumps(doc))
    source = {"path": "inputs/development_five.json", "sha256": ep.sha256(path),
              "role": doc["role"], "location": "Only the selected observations"}
    return root, request, source, doc


def mock_native(monkeypatch, calls, failed_first=False):
    def native(snapshot, request, output, root, *, audit, require_vss):
        assert audit is False and require_vss is False
        calls.append(output)
        if failed_first and len(calls) == 1:
            return {"status": "parameter_readback_failed", "reason": "physical capacity"}
        result = _native_success(snapshot, request, output, root, audit=audit)
        result.update(clearance_l_h=None, vss_l=None)
        return result
    monkeypatch.setattr(ep, "_run_development", native)


def fit(case, **kwargs):
    root, req, source, _ = case
    return ep.fit_mapping(reference_request=req, training_source=source,
        output_dir=root / "fit", project_root=root, max_evaluations=5, **kwargs)


def test_renal_input_preserved_and_only_nonrenal_scaled(evidence_case):
    root, req, _ = evidence_case
    blueprint = eb.assemble_evidence_blueprint(req, root)
    original = deepcopy(blueprint)
    native = ep.snapshot_for_mapping(blueprint, .3, 2.)
    assert blueprint == original
    for before, after in zip(blueprint["snapshot"]["Compounds"][0]["Processes"], native["Compounds"][0]["Processes"]):
        p = next(p for p in before["Parameters"] if p["Name"] == "Plasma clearance")
        q = next(p for p in after["Parameters"] if p["Name"] == "Plasma clearance")
        if before["InternalName"] == "KidneyClearance":
            assert p == q
        else:
            assert q["Value"] == p["Value"]*2


def test_no_renal_process_is_identical_to_original_mapping(evidence_case):
    root, req, _ = evidence_case
    b = eb.assemble_evidence_blueprint(req, root)
    b["snapshot"]["Compounds"][0]["Processes"] = [p for p in b["snapshot"]["Compounds"][0]["Processes"] if p["InternalName"] != "KidneyClearance"]
    assert ep.snapshot_for_mapping(b, .2, .5) == old.mapped_snapshot(b, .2, .5)


def test_fit_loss_sees_only_selected_five_rows_and_keeps_null_diagnostics(case, monkeypatch):
    calls = []; mock_native(monkeypatch, calls)
    result = fit(case)
    assert result["evaluation_count"] == result["actual_new_calls"] == len(calls) == 5
    assert result["warm_start_source"] is None
    assert result["decision"]["curve_metrics"]["point_count"] == 5
    assert result["decision"]["clearance_l_h"] is None
    assert result["decision"]["vss_l"] is None
    assert len(result["evaluations"][0]["predicted_at_observed_times"]) == 5
    for p in calls:
        flags = json.loads((p / "endpoints.json").read_text())
        assert flags["observations_used"] is False and flags["endpoint_estimator_uses_observations"] is False
    assert result["selected_prediction"]["path"].endswith("execution_0/prediction_locked.json")


def test_pause_escapes_optimizer_and_resume_replays_complete_scientific_failures(case, monkeypatch):
    calls = []; mock_native(monkeypatch, calls, failed_first=True)
    def halt():
        if len(calls) == 2:
            raise RuntimeError("project halt marker")
    with pytest.raises(ep.InfrastructurePause, match="halt marker"):
        fit(case, before_trial=halt)
    assert len(calls) == 2
    assert not (case[0] / "fit/mapping_locked.json").exists()
    result = fit(case)
    assert len(calls) == 5
    assert result["actual_new_calls"] == 3 and result["reused_calls"] == 2
    assert result["evaluations"][0]["status"] == "evaluation_failed"
    assert result["failed_evaluation_count"] == 1
    mapping = case[0] / result["mapping_path"]
    original_hash = ep.sha256(mapping)
    again = fit(case)
    assert again["actual_new_calls"] == 0 and again["reused_calls"] == 5
    assert len(calls) == 5 and again["evaluations"] == result["evaluations"]
    assert ep.sha256(mapping) == original_hash


def test_incomplete_invocation_retries_once_in_fresh_directory(case, monkeypatch):
    calls = []
    def interrupted(snapshot, request, output, root, **kwargs):
        calls.append(output)
        (output / "orphan-output.txt").write_text("diagnostics retained")
        raise KeyboardInterrupt("simulated host interruption")
    monkeypatch.setattr(ep, "_run_development", interrupted)
    with pytest.raises(KeyboardInterrupt):
        fit(case)
    first = calls[0]
    mock_native(monkeypatch, calls)
    result = fit(case)
    assert first.name == "execution_0" and calls[1].name == "execution_1"
    assert (first / "orphan-output.txt").read_text() == "diagnostics retained"
    assert result["infra_retry_calls"] == 1
    assert result["total_native_calls"] == 6


def test_repeated_infrastructure_failure_pauses_instead_of_spending_search_budget(case, monkeypatch):
    calls = []
    def failed(*args, **kwargs):
        calls.append(args[2])
        return {"status": "native_failed", "execution": {"stop_reason": "timeout"}}
    monkeypatch.setattr(ep, "_run_development", failed)
    with pytest.raises(ep.InfrastructurePause, match="single permitted retry"):
        fit(case)
    assert len(calls) == 2
    with pytest.raises(ep.InfrastructurePause):
        fit(case)
    assert len(calls) == 2
    assert not (case[0] / "fit/mapping_locked.json").exists()


def test_before_trial_halt_never_launches_native(case, monkeypatch):
    calls = []; mock_native(monkeypatch, calls)
    with pytest.raises(ep.InfrastructurePause, match="halt"):
        fit(case, before_trial=lambda: (_ for _ in ()).throw(ValueError("halt")))
    assert calls == []


@pytest.mark.parametrize("field", ["human", "dosing"])
def test_development_human_and_dose_must_match(case, monkeypatch, field):
    root, req, source, doc = case
    doc[field] = {**doc[field], **({"weight_kg": 99} if field == "human" else {"dose_mg": 99})}
    p = root / source["path"]; p.write_text(json.dumps(doc)); source["sha256"] = ep.sha256(p)
    with pytest.raises(ValueError, match="human/dosing"):
        fit(case)
    assert not (root / "fit").exists()


@pytest.mark.parametrize("kind", ["source", "parameters", "artifact", "code"])
def test_resume_rejects_changed_inputs_parameters_results_or_code(case, monkeypatch, kind):
    calls = []; mock_native(monkeypatch, calls)
    result = fit(case); root, req, source, doc = case
    if kind == "source":
        doc["observations"][0]["concentration_umol_per_l"] *= 2
        p = root/source["path"]; p.write_text(json.dumps(doc)); source["sha256"] = ep.sha256(p)
    elif kind == "parameters":
        p = root/f"fit/attempt_{result['decision']['evaluation']:02d}/parameters.json"; record=json.loads(p.read_text());record["clearance_scale"]=2;p.write_text(json.dumps(record))
    elif kind == "artifact":
        p=root/result['decision']['native_output_dir']/"result.json";p.write_text('{"status":"predicted"}')
    else:
        monkeypatch.setattr(ep, "_code_pins", lambda: {"different.py": "different"})
    with pytest.raises((ValueError, ep.InfrastructurePause)):
        fit(case)
    assert len(calls) == 5


def test_target_observations_cannot_be_training_source(case):
    root, req, source, _ = case
    req["forbidden_sources"] = [{"path": source["path"], "sha256": source["sha256"], "location": "Held-out target outcomes"}]
    with pytest.raises(ValueError, match="Comparison observations"):
        fit(case)
    assert not (root/"fit").exists()


def test_old_policy_mapping_is_rejected_before_native(case):
    root, request, _, _ = case
    p=root/"old_mapping.json";p.write_text(json.dumps({"version":"development-curve-pbpk-1.2","eligible":True}))
    with pytest.raises(ValueError, match="policy/version"):
        ep.predict_mapping(request=request,mapping_path=p,output_dir=root/"predict",project_root=root)
    assert not (root/"predict").exists()


def test_prediction_checks_new_policy_and_writes_explicit_estimator_flags(case, monkeypatch):
    calls=[];mock_native(monkeypatch,calls);result=fit(case)
    root,request,_,_=case
    monkeypatch.setattr(ep,"_run",_native_success)
    prediction=ep.predict_mapping(request=request,mapping_path=root/result["mapping_path"],output_dir=root/"target",project_root=root)
    assert prediction["status"]=="predicted"
    assert prediction["clearance_policy"]==ep.CLEARANCE_POLICY
    assert prediction["endpoint_estimator_uses_observations"] is False
    assert (root/"target/request.json").is_file() and (root/"target/result.json").is_file()
    d=json.loads((root/"target/endpoints.json").read_text())
    assert d["observations_used"] is False and d["development_curve_used_for_parameter_calibration"] is True


def test_completed_mapping_survives_unselected_success_cleanup(case, monkeypatch):
    calls=[];mock_native(monkeypatch,calls,failed_first=True)
    result=fit(case);root,request,_,_=case
    assert 1 in result["retained_evaluations"]
    assert result["disposable_successful_evaluations"]
    checksum=ep.sha256(root/result["mapping_path"])
    for number in result["disposable_successful_evaluations"]:
        shutil.rmtree(root/f"fit/attempt_{number:02d}")
    again=fit(case)
    assert again["completed_fit_reused"] is True and again["actual_new_calls"]==0
    assert len(calls)==5 and ep.sha256(root/result["mapping_path"])==checksum
    assert (root/"fit/attempt_01/execution_0/result.json").exists()
    monkeypatch.setattr(ep,"_run",_native_success)
    predicted=ep.predict_mapping(request=request,mapping_path=root/result["mapping_path"],output_dir=root/"target",project_root=root)
    assert predicted["status"]=="predicted"


def test_checkpoint_io_failure_does_not_consume_remaining_budget(case, monkeypatch):
    calls=[];mock_native(monkeypatch,calls)
    original=ep.save
    def disk_failure(path,data):
        if path.name=="completion.json":
            raise OSError("No space left on device")
        return original(path,data)
    monkeypatch.setattr(ep,"save",disk_failure)
    with pytest.raises(ep.InfrastructurePause,match="No space left"):
        fit(case)
    assert len(calls)==1
