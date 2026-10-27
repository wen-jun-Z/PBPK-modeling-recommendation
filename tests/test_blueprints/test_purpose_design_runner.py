"""Runner trust boundaries with synthetic sources and the real exchange solver.

Only native physiology export is mocked in numerical/lifecycle checks. All
temporary files stay under this project's purpose_design_tests cache.
"""
from copy import deepcopy
import math
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
import design_evidence_pbpk as runner
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
from test_evidence_decision import evidence, fallback, record, request
from test_perfusion_model import physiology as physiology_fixture


@pytest.fixture
def workspace(monkeypatch):
    root = PROJECT / "runtime/cache/purpose_design_tests" / uuid.uuid4().hex
    root.mkdir(parents=True)
    monkeypatch.setattr(runner, "ROOT", root)
    # Freeze stand-in code artifacts without changing the real imported code.
    for name in ("scripts/design_evidence_pbpk.py",
                 "src/pbpk_kb/blueprint_policy/purpose_design.py",
                 "src/pbpk_kb/blueprint_policy/evidence_decision.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((PROJECT / name).read_bytes())
    monkeypatch.setattr(runner, "__file__", str(root / "scripts/design_evidence_pbpk.py"))
    monkeypatch.setattr(runner.engine, "_code_pins", lambda _: {})
    def no_fit(*args, **kwargs):
        pytest.fail("Purpose construction must not fit any clinical curve")
    monkeypatch.setattr(runner.engine, "fit_mapping", no_fit)
    yield root
    shutil.rmtree(root)


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.save(path, value)
    return {**runner.ref(path), "location": "synthetic fixture"}


def document(root, *, predicted=True, ranged=True, renal=True):
    req = request()
    req["human"]["weight_kg"] = 80.
    req["dosing"]["dose_mg"] = 20.
    req["target"].update(endpoints=["auc", "cmax"], primary_endpoint="auc", window_end_h=12.)
    record(req, "molecular_weight")["value"] = 500.
    record(req, "total_clearance")["value"] = 2.
    if predicted:
        req["evidence"] = [row for row in req["evidence"] if row["parameter"] != "total_clearance"]
        cl = fallback()[1]
        cl["value"] = 2.
        req["evidence"].append(cl)
    if renal:
        record(req, "elimination_scope")["value"] = "hepatic_and_renal"
        req["evidence"].append(evidence("renal_clearance", .5, "ml/min/kg", context={
            "species": "human", "matrix": "plasma", "route": "intravenous"}))
    source = put(root, "sources/static.json", {"kind": "synthetic", "not_clinical_data": True})
    for row in req["evidence"]:
        row["source"] = {**source, "location": row["parameter"]}
    req["evidence_ledger"] = put(root, "sources/ledger.json", {"evidence": req["evidence"]})
    ranges = []
    if predicted and ranged:
        ranges = [{"parameter": "total_clearance", "lower": 1., "upper": 3.,
                   "unit": "ml/min/kg", "basis": "stress_assumption",
                   "source": put(root, "sources/range.json", {"kind": "declared sensitivity assumption"})}]
    return {"request": req, "fallback_evidence": [], "ranges": ranges,
            "endpoint_limits": {"auc": {"metric": "fold", "max_change": 100.},
                                "cmax": {"metric": "fold", "max_change": 100.}}}


def freeze(root, doc):
    source = root / "design_input.json"
    runner.save(source, doc)
    output = root / "experiments/blueprint_design/purpose_test"
    runner.prepare(source, output)
    return output


def real_physiology(monkeypatch):
    calls = []
    physiology = physiology_fixture.__wrapped__()
    def export(req, folder, project_root, guard):
        guard()
        calls.append(deepcopy(req))
        runner.save(folder / "physiology.json", physiology)
        return deepcopy(physiology)
    monkeypatch.setattr(runner.engine, "_physiology", export)
    return calls, physiology


def mapping_fixture(root, req, monkeypatch, *, training=None):
    training = training or put(root, "fit/training.json", {"kind": "synthetic development only"})
    put(root, "fit/input.json", {"request": req})
    path = root / "fit/mapping.json"
    runner.save(path, {"eligible": True, "variant": "flow_exchange", "training_source": training,
                       "decision": {"log_parameters": [math.log(3.), 0., 0.]}})
    # This boundary is tested independently of the old engine's fit archive
    # validator; no fake optimizer output or real fitting is needed here.
    validated = []
    monkeypatch.setattr(runner.engine, "_validated", lambda *args: validated.append(args))
    return runner.ref(path), validated


def test_source_hash_and_ledger_are_checked_before_planning(workspace, monkeypatch):
    doc = document(workspace)
    runner.checked_input(doc)
    (workspace / "sources/static.json").write_text('{"changed": true}')
    with pytest.raises(ValueError):
        runner.checked_input(doc)


@pytest.mark.parametrize("change", ["value", "source"])
def test_request_cannot_silently_diverge_from_source_ledger(workspace, change):
    doc = document(workspace)
    if change == "value":
        record(doc["request"], "fraction_unbound")["value"] = .2
    else:
        record(doc["request"], "fraction_unbound")["source"] = put(workspace, "different.json", {"different": True})
    with pytest.raises(ValueError, match="original source ledger"):
        runner.checked_input(doc)


def test_ledger_order_is_not_a_false_input_change(workspace):
    doc = document(workspace)
    doc["request"]["evidence"].reverse()
    assert runner.checked_input(doc)


@pytest.mark.parametrize("copy_source", [False, True])
def test_forbidden_source_rejected_by_path_or_content_without_decoding(workspace, monkeypatch, copy_source):
    doc = document(workspace)
    forbidden = put(workspace, "heldout.json", {"do_not_decode": True})
    doc["request"]["forbidden_sources"] = [forbidden]
    used = forbidden
    if copy_source:
        copied = workspace / "renamed_outcome.json"
        copied.write_bytes((workspace / "heldout.json").read_bytes())
        used = {**runner.ref(copied), "location": "renamed"}
    doc["ranges"][0]["source"] = used
    original = runner.read
    def watched(path):
        if Path(path).name in {"heldout.json", "renamed_outcome.json"}:
            pytest.fail("Forbidden outcome was decoded before rejection")
        return original(path)
    monkeypatch.setattr(runner, "read", watched)
    with pytest.raises(ValueError, match="forbidden outcome"):
        runner.checked_input(doc)


@pytest.mark.parametrize("key", ["observations", "target_curve", "clinical_auc"])
def test_outcomes_are_not_runner_input_fields(workspace, key):
    doc = document(workspace)
    doc[key] = [1., 2.]
    with pytest.raises(ValueError, match="outcomes are not design inputs"):
        runner.checked_input(doc)


@pytest.mark.parametrize("change", ["fu", "partition", "provenance"])
def test_old_calibration_cannot_be_reused_after_nominal_drug_change(workspace, monkeypatch, change):
    doc = document(workspace)
    ref, validated = mapping_fixture(workspace, doc["request"], monkeypatch)
    doc["calibration_mapping"] = ref
    selected = deepcopy(doc["request"])
    if change == "fu":
        record(selected, "fraction_unbound")["value"] = .2
    elif change == "partition":
        selected["partition_method"] = "RodgersRowland"
    else:
        record(selected, "total_clearance")["source"] = put(workspace, "new_prediction.json", {"new_model": True})
    with pytest.raises(ValueError, match="nominal drug inputs differ"):
        runner.mapping_parameters(doc, selected)
    assert len(validated) == 1


def test_fixed_mapping_allows_new_subject_and_regimen_but_never_refits(workspace, monkeypatch):
    doc = document(workspace)
    ref, validated = mapping_fixture(workspace, doc["request"], monkeypatch)
    doc["calibration_mapping"] = ref
    selected = deepcopy(doc["request"])
    selected["human"]["weight_kg"] = 65.
    selected["dosing"].update(dose_mg=30., infusion_minutes=60.)
    selected["target"]["window_end_h"] = 24.
    got = runner.mapping_parameters(doc, selected)
    assert got["log_parameters"] == [math.log(3.), 0., 0.]
    assert got["scenario_refitting"] is False and got["development_curve_used"] is True
    assert len(validated) == 1


@pytest.mark.parametrize("nested", [False, True])
def test_current_forbidden_sources_override_old_fit_training_eligibility(workspace, monkeypatch, nested):
    doc = document(workspace)
    forbidden = put(workspace, "heldout.json", {"do_not_decode": True})
    training = put(workspace, "fit/training.json", {"raw_source": forbidden}) if nested else forbidden
    ref, _ = mapping_fixture(workspace, doc["request"], monkeypatch, training=training)
    doc["calibration_mapping"] = ref
    selected = deepcopy(doc["request"])
    selected["forbidden_sources"] = [forbidden]
    original = runner.read
    def watched(path):
        if Path(path).name == "heldout.json":
            pytest.fail("Forbidden training outcome was decoded")
        return original(path)
    monkeypatch.setattr(runner, "read", watched)
    with pytest.raises(ValueError, match="currently forbidden"):
        runner.mapping_parameters(doc, selected)


@pytest.mark.parametrize("parameter", ["fraction_unbound", "renal_clearance", "molecular_weight"])
def test_scenario_cannot_modify_measured_or_unranged_parameters(workspace, parameter):
    output = freeze(workspace, document(workspace))
    _, protocol, _ = runner.load(output)
    scenario = deepcopy(protocol["decision"]["scenarios"][0])
    scenario["values"][parameter] = .123
    with pytest.raises(ValueError, match="only change explicitly ranged predictions"):
        runner.selected_scenario(protocol, scenario, output / "reject_test")


def test_scenario_keeps_measured_evidence_and_declares_range_as_nonmeasurement(workspace):
    doc = document(workspace)
    output = freeze(workspace, doc)
    _, protocol, _ = runner.load(output)
    scenario = next(row for row in protocol["decision"]["scenarios"] if row["values"]["total_clearance"] == 1.)
    selected = runner.selected_scenario(protocol, scenario, output / "scenario_test")
    for name in ("fraction_unbound", "renal_clearance", "molecular_weight", "logp"):
        assert record(selected, name) == record(protocol["decision"]["selected_request"], name)
    changed = record(selected, "total_clearance")
    assert changed["basis"] == "predicted" and changed["value"] == 1.
    provenance = runner.read(changed["source"]["path"])
    assert provenance["kind"] == "sensitivity_scenario_not_observed"
    assert provenance["range_is_probability_interval"] is False
    assert provenance["nominal_evidence"]["value"] == 2.
    assert runner.read(selected["evidence_ledger"]["path"])["evidence"] == selected["evidence"]


def test_real_solver_interface_units_mass_balance_and_fixed_renal(workspace, monkeypatch):
    doc = document(workspace)
    output = freeze(workspace, doc)
    _, protocol, _ = runner.load(output)
    protocol["calibration"]["log_parameters"] = [math.log(3.), 0., 0.]
    calls, physiology = real_physiology(monkeypatch)
    scenario = protocol["decision"]["scenarios"][0]
    result = runner.simulate_scenario(protocol, scenario, output / "interface_test", lambda: None)
    assert len(calls) == 1
    # 80 kg: fixed renal=.5*80*.06=2.4 L/h; nonrenal=(2-.5)*80*.06.
    assert result["effective_total_clearance_l_h"] == pytest.approx(2.4 + 7.2 * 3.)
    assert result["mass_balance"]["maximum_relative_error"] < 1e-9
    direct = runner.engine._simulate(physiology, calls[0], np.array(result["curve"]["time_h"]),
                                     "flow_exchange", runner.engine.PARAMETERS["flow_exchange"],
                                     [math.log(3.), 0., 0.])
    conc = np.array(result["curve"]["concentration_mg_l"])
    np.testing.assert_allclose(conc, direct["concentration_mg_l"], rtol=1e-10, atol=1e-12)
    assert np.all(conc >= 0.) and np.any(conc > 0.)
    assert result["endpoints"]["cmax"]["value"] == pytest.approx(conc.max(), rel=1e-10)
    assert result["endpoints"]["cmax"]["unit"] == "mg/l"
    assert result["endpoints"]["auc"]["unit"] == "mg*h/l"
    assert result["endpoint_estimator_uses_observations"] is False


@pytest.mark.parametrize("predicted,status", [(True, "needs_evidence"), (False, "not_assessed")])
def test_absent_ranges_never_become_a_stability_claim(workspace, monkeypatch, predicted, status):
    output = freeze(workspace, document(workspace, predicted=predicted, ranged=False))
    calls, _ = real_physiology(monkeypatch)
    completed = runner.execute(output)
    assert completed["status"] == "complete" and completed["failed"] == 0
    result = runner.read(output / "result.json")
    assert result["assessment"]["status"] == status
    assert result["assessment"]["clinical_validation"] is False
    assert result["target_observations_used"] is False
    assert len(calls) == 1
    assert runner.execute(output) == completed
    assert len(calls) == 1  # Completed runs do not submit native work again.


def test_purpose_endpoint_override_reaches_real_request_and_summary(workspace, monkeypatch):
    doc = document(workspace, ranged=False)
    doc["purpose"] = {"endpoints": ["c_last"]}
    doc["endpoint_limits"] = {"c_last": {"metric": "fold", "max_change": 2.}}
    output = freeze(workspace, doc)
    calls, _ = real_physiology(monkeypatch)
    runner.execute(output)
    assert calls[0]["target"]["endpoints"] == ["c_last"]
    assert calls[0]["target"]["primary_endpoint"] == "c_last"
    got = runner.read(output / "result.json")
    assert set(got["predictions"][0]["endpoints"]) == {"c_last"}
    assert got["predictions"][0]["endpoints"]["c_last"]["value"] > 0.


def test_failed_scenario_is_retained_and_prevents_support(workspace, monkeypatch):
    output = freeze(workspace, document(workspace))
    real_physiology(monkeypatch)
    original = runner.simulate_scenario
    def fail_one(protocol, scenario, folder, guard):
        if scenario["scenario_id"] == "scenario_001":
            raise ValueError("Synthetic numerical domain failure")
        return original(protocol, scenario, folder, guard)
    monkeypatch.setattr(runner, "simulate_scenario", fail_one)
    assert runner.execute(output)["failed"] == 1
    result = runner.read(output / "result.json")
    assert len(result["predictions"]) == 3
    assert result["assessment"]["status"] == "needs_evidence"
    failed = next(row for row in result["predictions"] if row["status"] == "failed")
    assert "Synthetic numerical" in failed["reason"]
    assert any(path.endswith("prediction.json") for path in failed["artifacts"])


@pytest.mark.parametrize("tamper", ["physiology", "inline_prediction", "prediction_artifact", "result"])
def test_resume_rejects_modified_prediction_or_artifact(workspace, monkeypatch, tamper):
    output = freeze(workspace, document(workspace, ranged=False))
    calls, _ = real_physiology(monkeypatch)
    runner.execute(output)
    if tamper == "inline_prediction":
        state = runner.read(output / "state.json")
        state["predictions"][0]["endpoints"]["auc"]["value"] *= 2.
        runner.save(output / "state.json", state)
    else:
        path = (output / "result.json" if tamper == "result" else output / "models/baseline" /
                ("physiology.json" if tamper == "physiology" else "prediction.json"))
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        runner.execute(output)
    assert len(calls) == 1


@pytest.mark.parametrize("tamper", ["input", "source", "code"])
def test_prepare_locks_direct_input_source_and_code(workspace, tamper):
    output = freeze(workspace, document(workspace))
    path = workspace / {"input": "design_input.json", "source": "sources/static.json",
                        "code": "src/pbpk_kb/blueprint_policy/purpose_design.py"}[tamper]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        runner.load(output)


@pytest.mark.parametrize("relative", ["purpose_outside", "experiments/purpose_wrong_parent",
                                      "experiments/blueprint_design/unprefixed"])
def test_output_is_confined_to_dedicated_immediate_design_directory(workspace, relative):
    doc = document(workspace)
    source = workspace / "design_input.json"
    runner.save(source, doc)
    output = workspace / relative
    with pytest.raises(ValueError, match="immediate"):
        runner.prepare(source, output)
    assert not output.exists()


def test_interrupted_native_scenario_requires_inspection_and_no_resubmission(workspace, monkeypatch):
    output = freeze(workspace, document(workspace))
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("Synthetic infrastructure interruption")
    monkeypatch.setattr(runner.engine, "_physiology", interrupted)
    with pytest.raises(RuntimeError, match="infrastructure"):
        runner.execute(output)
    state = runner.read(output / "state.json")
    assert state["status"] == "paused" and state["incomplete_scenario"] == "baseline"
    with pytest.raises(ValueError, match="Interrupted native scenario retained"):
        runner.execute(output)
    assert len(calls) == 1


@pytest.mark.parametrize("dependency", ["training", "retained"])
def test_resume_revalidates_transitive_locked_mapping_dependencies(workspace, dependency):
    """Use the actual old archive validator, without running an optimizer."""
    doc = document(workspace)
    req = EvidenceBlueprintRequest.model_validate(doc["request"]).model_dump(mode="json", exclude_none=True)
    training = put(workspace, "fit/training.json", {"kind": "synthetic archive fixture"})
    retained = put(workspace, "fit/physiology.json", {"kind": "synthetic retained artifact"})
    source_pins = {training["path"]: training["sha256"]}
    fixed = {"request": req, "variant": "flow_exchange", "source_pins": source_pins,
             "code_pins": {}, "training_source": training,
             "clearance_input": runner.engine.clearance_inputs(req)}
    input_ref = put(workspace, "fit/input.json", fixed)
    decision = {"index": 0, "eligible": True, "objective": 1., "log_parameters": [0., 0., 0.]}
    progress = {"status": "locked", "starts": [decision], "linear_solver_calls": 0,
                "failed_evaluation_count": 0, "cost_count_exact": 0}
    put(workspace, "fit/progress.json", progress)
    mapping = {"version": runner.engine.VERSION, "input_sha256": input_ref["sha256"],
               "variant": "flow_exchange", "optimizer_config": runner.engine.CONFIG,
               "drug_signature": runner.engine._drug_signature(req), "source_pins": source_pins,
               "code_pins": {}, "retained_artifact_pins": {retained["path"]: retained["sha256"]},
               "construction_partition_method": req["partition_method"], "training_source": training,
               "reference_human": req["human"], "reference_dosing": req["dosing"],
               "clearance_input": fixed["clearance_input"],
               "active_parameters": list(runner.engine.PARAMETERS["flow_exchange"]),
               "decision": decision, "eligible": True,
               **{k: v for k, v in progress.items() if k != "status"}}
    mapping_ref = put(workspace, "fit/mapping.json", mapping)
    doc["calibration_mapping"] = {k: mapping_ref[k] for k in ("path", "sha256")}
    output = freeze(workspace, doc)
    runner.load(output)
    path = workspace / (training if dependency == "training" else retained)["path"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        runner.load(output)
