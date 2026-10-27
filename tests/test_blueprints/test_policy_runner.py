"""Evidence-policy orchestration isolation and comparison contracts; no native run."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
import run_blueprint_evidence_policy as runner
from test_evidence_decision import request as example_request, fallback as example_fallback


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    runner.save(path,value)
    return runner.ref(path)


@pytest.fixture
def batch(tmp_path,monkeypatch):
    if not tmp_path.resolve().is_relative_to(ROOT):
        pytest.skip("Use a project-local --basetemp")
    base,run=tmp_path/"frozen",tmp_path/"prepared"
    base.mkdir()
    monkeypatch.setattr(runner,"BASE",base)
    def owned(path):
        if Path(path)!=run:raise ValueError("This fixture owns only its prepared directory")
        return run
    monkeypatch.setattr(runner,"own_run",owned)
    source=write(base/"source.json",{"kind":"synthetic source","observed_target":False})
    target_observation=base/"forbidden_target_values.json"
    target_observation.write_text("FORBIDDEN TARGET: decoding this invalid JSON must fail")
    forbidden={**runner.ref(target_observation),"location":"Target outcomes; not inputs"}
    request=example_request()
    for record in request["evidence"]:record["source"]={**source,"location":record["parameter"]}
    request["evidence_ledger"]={**source,"location":"original structured evidence"}
    request["forbidden_sources"]=[forbidden]
    target=deepcopy(request);target["dosing"]["dose_mg"]*=2
    rf=deepcopy(request)
    backup=example_fallback()[0];backup["source"]={**source,"location":"Frozen RF fu prediction"}
    rf["evidence"]=[backup if record["parameter"]=="fraction_unbound" else record for record in rf["evidence"]]
    development={"compound_id":request["compound_id"],"compound_name":request["compound_name"],
        "study_id":"synthetic_development","role":"calibration_data_reconstruction","matrix":"plasma",
        "time_unit":"min","unit":"umol/l","molecular_weight_g_mol":300.,
        "human":deepcopy(request["human"]),"dosing":deepcopy(request["dosing"]),
        "observations":[{"time_min":float(i+1)*30,"concentration_umol_per_l":1./(i+1),
            "hidden_full_concentration":99999} for i in range(7)],
        "raw_observations":[{"target_concentration":99999}],
        "training_auc_from_full_curve":99999,"warm_start_parameters":{"lipophilicity":99999},
        "sampling":{"full_curve_source":forbidden}}
    five=deepcopy(development);five["observations"]=five["observations"][:5]
    case={"case_id":"synthetic_case","compound_id":request["compound_id"],"drug":"Synthetic",
        "study_id":"synthetic_target","cluster_id":"synthetic_target","transfer_group":"primary",
        "reference_source":forbidden,"source_outcomes_must_not_decode":forbidden,
        "target_request_metadata":{"human":target["human"],"dosing":target["dosing"],"window_end_h":24.},
        "nca_policy":{"sampling_time_h":[.5,1.,2.,4.,8.]},
        "requests":{"evidence":{"development":write(base/"development_request.json",request),
                                  "target":write(base/"target_request.json",target)},
                    "rf":{"development":write(base/"rf_request.json",rf)}},
        "sampling_sources":{"full":write(base/"full.json",development),"five":write(base/"five.json",five)}}
    pins={str(path.relative_to(ROOT)):runner.sha256(path) for path in base.iterdir() if path.is_file()}
    write(base/"protocol.json",{"source_pins":pins,"cases":[case]})
    return run,case,request,rf,development


def prepare(batch):
    run=batch[0]
    result=runner.prepare(run,workers=2)
    return runner.read(run/"protocol.json"),result


def test_clean_training_keeps_only_supplied_observations_no_hidden_full_curve(batch):
    original=deepcopy(batch[4]);five=deepcopy(original);five["observations"]=five["observations"][:5]
    cleaned=runner.clean_training(five)
    assert len(cleaned["observations"])==5 and len(original["observations"])==7
    assert set(cleaned["observations"][0])=={"time_min","concentration_umol_per_l"}
    assert not {"raw_observations","warm_start_parameters","training_auc_from_full_curve","sampling"}&cleaned.keys()
    cleaned["human"]["weight_kg"]=123.
    assert batch[4]["human"]["weight_kg"]==70.


def test_conflict_is_same_priority_and_audit_truth_not_visible(batch):
    request,rf=batch[2:4];before=deepcopy(request)
    rows,fallback,audit=runner.scenario_evidence(request,rf,"conflicting_fu","synthetic_case")
    fu=[row for row in rows if row["parameter"]=="fraction_unbound"]
    assert len(fu)==2 and {row["basis"] for row in fu}=={"measured"}
    assert sorted(row["value"] for row in fu)==[.05,.1]
    assert all(row["context"]["species"]=="human" for row in fu)
    assert all(not {"source","notes","synthetic","perturbation","expected_ready"}&row.keys() for row in rows+fallback)
    assert audit["synthetic"] and audit["perturbation"]=="conflicting_fu"
    assert next(row for row in audit["primary_sources"] if row["parameter"]=="fraction_unbound")["value"]==.1
    assert request==before


def test_wrong_species_is_visible_context_and_original_stays_only_in_audit(batch):
    request,rf=batch[2:4]
    rows,fallback,audit=runner.scenario_evidence(request,rf,"wrong_species_fu","synthetic_case")
    assert next(row for row in rows if row["parameter"]=="fraction_unbound")["context"]["species"]=="rat"
    assert fallback[0]["context"]["species"]=="human"
    assert next(row for row in audit["primary_sources"] if row["parameter"]=="fraction_unbound")["context"]["species"]=="human"
    assert next(row for row in request["evidence"] if row["parameter"]=="fraction_unbound")["context"]["species"]=="human"


def test_prepare_complete_matrix_without_decoding_target_oracle(batch):
    from pbpk_kb.blueprints.development_curve_blueprint import _training_document
    protocol,counts=prepare(batch)
    assert counts["decisions"]==36 and counts["logical_units"]==72
    assert len({unit["id"] for unit in protocol["units"]})==72
    assert protocol["constraints"]["target_outcomes_read_only_after_all_attempts_locked"]
    assert protocol["constraints"]["aliases_are_not_independent_experiments"]
    for computation in protocol["computations"].values():
        document=runner.checked(computation["training_source"])
        request=runner.checked(computation["request"])
        assert computation["training_source"]["role"]=="calibration_data_reconstruction"
        assert _training_document(computation["training_source"],request,ROOT)==document
        assert len(document["observations"])==(5 if computation["sampling"]=="five" else 7)
        assert "raw_observations" not in document and "warm_start_parameters" not in document
        assert "source_outcomes_must_not_decode" not in computation.get("case",{})
        if computation["kind"]=="classical":
            physchem=runner.checked(computation["case"]["physchem"])
            assert physchem=={"identity":{"molecular_weight_g_mol":300.}}
            evidence=runner.checked(computation["request"])["evidence"]
            assert {e["parameter"] for e in evidence}<={"total_clearance","hepatic_clearance","renal_clearance","elimination_scope"}
    assert all(not {"scenario","synthetic","perturbation","expected_ready"}&runner.checked(c["request"]).keys()
               for c in protocol["computations"].values())


def test_native_and_classical_oracles_follow_required_inputs_not_model_preference(batch):
    protocol,_=prepare(batch)
    decisions={(d["scenario"],d["method"]):d for d in protocol["decisions"]}
    for method in runner.METHODS:
        assert not decisions["unsupported_ddi",method]["expected_ready"]
        assert decisions["unsupported_ddi",method]["status"]=="blocked"
        assert decisions["available",method]["expected_ready"]
        for scenario in ("conflicting_fu","missing_required_pbpk_formula"):
            expected=method in runner.CLASSICAL
            assert decisions[scenario,method]["expected_ready"] is expected
            if expected:assert decisions[scenario,method]["status"]=="ready"
    assert decisions["missing_fu","evidence_policy"]["status"]=="ready"
    assert decisions["wrong_species_fu","evidence_policy"]["status"]=="ready"
    assert decisions["missing_fu","fixed_template"]["status"]=="blocked"
    assert decisions["conflicting_fu","evidence_policy"]["status"]=="blocked"
    # This is the deliberately unsafe conflict-handling control, not a missing experiment.
    assert decisions["conflicting_fu","no_conflict_guard"]["status"]=="ready"
    assert not decisions["conflicting_fu","no_conflict_guard"]["expected_ready"]


def test_content_addressed_views_hide_case_truth_but_keep_auditable_source_records(batch):
    protocol,_=prepare(batch)
    for decision in protocol["decisions"]:
        view=decision["visible_evidence"]
        document=runner.checked(view)
        assert set(document)=={"records"}
        assert len(Path(view["path"]).stem)==64
        for selected in decision["decision"]["selected_evidence"]:
            if "source" in selected:
                assert selected["source"]["path"]==view["path"]
                assert selected["source"]["sha256"]==view["sha256"]
                index=int(selected["source"]["location"].removeprefix("Record "))-1
                assert document["records"][index]==runner.visible_record(selected)
    assert len(protocol["audit_only_not_passed_to_methods"])==6


def test_same_numeric_inputs_alias_provenance_variants_but_different_inputs_do_not(batch):
    request=deepcopy(batch[2]);target=deepcopy(request)
    training={"path":"input/five.json","sha256":"a"*64}
    signature=lambda req,doc=training,tgt=target,allowed=("lipophilicity","clearance_scale"):runner.digest(
        runner.computation_signature(req,doc,tgt,allowed))
    original=signature(request)
    provenance=deepcopy(request)
    for row in provenance["evidence"]:row["source"]={"path":"different/source.json","sha256":"b"*64,"location":"another reference"}
    provenance["evidence_ledger"]={"path":"another/ledger.json","sha256":"c"*64}
    assert signature(provenance)==original
    changed=deepcopy(request);next(e for e in changed["evidence"] if e["parameter"]=="fraction_unbound")["value"]*=.5
    assert signature(changed)!=original
    assert signature(request,{"path":"input/full.json","sha256":"d"*64})!=original
    different_target=deepcopy(target);different_target["dosing"]["dose_mg"]*=2
    assert signature(request,tgt=different_target)!=original
    assert signature(request,allowed=("clearance_scale",))!=original


def test_aliases_preserve_each_policy_audit_and_separate_sampling(batch):
    protocol,_=prepare(batch)
    units={(u["scenario"],u["sampling"],u["method"]):u for u in protocol["units"]}
    for sampling in runner.SAMPLING:
        for method in runner.CLASSICAL:
            baseline=units["available",sampling,method]["computation_id"]
            for scenario in ("missing_fu","conflicting_fu","wrong_species_fu","missing_required_pbpk_formula"):
                assert units[scenario,sampling,method]["computation_id"]==baseline
                assert units[scenario,sampling,method]["decision_id"]!=units["available",sampling,method]["decision_id"]
        assert units["available",sampling,"evidence_policy"]["computation_id"]==units["available",sampling,"fixed_template"]["computation_id"]
        assert units["available",sampling,"evidence_policy"]["computation_id"]!=units["available",sampling,"rf_fill"]["computation_id"]
    for method in runner.METHODS:
        assert units["available","full",method]["computation_id"]!=units["available","five",method]["computation_id"]


def test_prepare_never_overwrites_existing_batch_and_source_changes_are_detected(batch):
    protocol,_=prepare(batch)
    with pytest.raises(FileExistsError):runner.prepare(batch[0])
    path=ROOT/next(iter(protocol["computations"].values()))["training_source"]["path"]
    path.write_text('{}')
    with pytest.raises(ValueError,match="Changed"):runner.verify(protocol["source_pins"])


@pytest.mark.parametrize("scientific",[True,False])
def test_classical_scientific_failure_is_recorded_but_integrity_error_pauses(batch,monkeypatch,scientific):
    import blueprint_validation_classical as classical
    import report_blueprint_evidence_policy as reporting
    protocol,_=prepare(batch);run=batch[0]
    identity,config=next((k,v) for k,v in protocol["computations"].items() if v["kind"]=="classical")
    protocol["computations"]={identity:config};protocol["workers"]=1
    runner.save(run/"protocol.json",protocol)
    state=runner.read(run/"state.json");state["protocol"]=runner.ref(run/"protocol.json");runner.save(run/"state.json",state)
    # The monitor is unrelated to the science/infrastructure distinction; no long-lived thread needed.
    monkeypatch.setattr(runner.Controller,"watch",lambda self:self.closed.wait(.1))
    def fit_predict(**kwargs):
        if scientific:
            write(kwargs["output_dir"]/"fit.json",{"status":"failed","starts":[{"error":"synthetic nonfinite fit"}]})
            raise RuntimeError("No finite subset-only classical fit; all attempts retained")
        raise ValueError("Frozen source integrity changed")
    monkeypatch.setattr(classical,"fit_predict",fit_predict)
    reported=[]
    def report(folder,**kwargs):
        state=runner.read(folder/"state.json")
        assert state["all_attempts_locked_before_scoring"] is True
        assert state["computations"][identity]["status"]=="failed"
        reported.append(True);write(folder/"results.json",{"status":"synthetic report only"})
    monkeypatch.setattr(reporting,"report",report)
    if scientific:
        runner.execute(run)
        state=runner.read(run/"state.json")
        assert state["status"]=="complete" and reported==[True]
        assert runner.checked(state["computations"][identity]["fit"])["status"]=="failed"
    else:
        with pytest.raises(runner.Paused,match="integrity changed"):runner.execute(run)
        assert runner.read(run/"state.json")["status"]=="paused"
        assert reported==[]
