"""Small audited batches exercise the production 100-compound controller.

Native execution is mocked; purpose planning and assessment are real. Pytest's
basetemp is explicitly placed in project runtime/cache by the invoking command.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
import design_evidence_pbpk as single
import run_purpose_hundred as hundred
from pbpk_kb.blueprints.source_workflow import save, sha256
from test_evidence_decision import request, fallback


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    save(path, value)
    return {"path": str(path.relative_to(root)), "sha256": sha256(path)}


@pytest.fixture
def batch(tmp_path, monkeypatch):
    assert tmp_path.is_relative_to(PROJECT / "runtime/cache"), "Tests must use project-local --basetemp"
    root = tmp_path
    monkeypatch.setattr(single, "ROOT", root)
    monkeypatch.setattr(single.engine, "_code_pins", lambda _: {})
    for name in hundred.CODE:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = PROJECT / name
        destination.write_bytes(source.read_bytes() if source.exists() else b"# isolated fixture artifact\n")
    manuscript = root / "results/manuscript/main.tex"
    manuscript.parent.mkdir(parents=True)
    manuscript.write_text("Frozen fixture publication.\n")
    run = root / "experiments/blueprint_design/purpose_hundred_fixture"
    static = {**put(root, "sources/static.json", {"kind": "synthetic physical chemistry"}), "location": "fixture"}
    ranges_source = {**put(root, "sources/ranges.json", {"basis": "declared stress assumptions"}), "location": "CL"}
    cases = []
    for i, smiles in enumerate(("CCO", "CCC", "CCN")):
        cid, compound = f"drug_{i}", f"synthetic:{i}"
        req = request()
        req.update(compound_id=compound, compound_name=cid)
        req["target"].update(endpoints=["auc"], primary_endpoint="auc")
        req["evidence"] = [row for row in req["evidence"] if row["parameter"] != "total_clearance"] + [fallback()[1]]
        for row in req["evidence"]:
            row.update(compound_id=compound, source=deepcopy(static))
        req["evidence_ledger"] = {**put(root, f"inputs/{cid}/ledger.json", {"evidence": req["evidence"]}), "location": "evidence"}
        evaluation = put(root, f"evaluation/{cid}.json", {"held_out": True, "auc": 11. + i})
        document = {"request": req, "fallback_evidence": [],
                    "ranges": [{"parameter": "total_clearance", "lower": 3., "upper": 12.,
                                "unit": "ml/min/kg", "basis": "stress_assumption", "source": ranges_source}],
                    "endpoint_limits": {"auc": {"metric": "fold", "max_change": 2.1}}}
        cases.append({"case_id": cid, "compound_id": compound, "identity_smiles": smiles,
                      "drug_name": cid, "cohort_role": "development_extension",
                      "evaluation_scope": "aggregated_human_IV_PK_parameters",
                      "input": put(root, f"inputs/{cid}/input.json", document), "evaluation": evaluation})
    manifest = put(root, "manifest.json", {"version": hundred.MANIFEST_VERSION, "cases": cases})
    calls, score_calls = [], []

    def simulate(protocol, scenario, folder, guard):
        guard()
        calls.append((protocol["input"]["request"]["compound_id"], scenario["scenario_id"]))
        (folder / "native_audit.txt").write_text("synthetic native output\n")
        clearance = scenario["values"]["total_clearance"]
        return {"scenario_id": scenario["scenario_id"], "status": "predicted",
                "endpoints": {"auc": {"value": 60. / clearance, "unit": "mg*h/l"}},
                "curve": {"time_h": [0., 1., 2.], "concentration_mg_l": [0., 1., .5]}}

    def scorer(protocol, outcomes, project_root):
        state = json.loads((run / "state.json").read_text())
        assert state["all_predictions_locked_before_scoring"] is True
        assert len(state["outcomes"]) == len(protocol["cases"]) == 3
        score_calls.append(len(outcomes))
        labels = [json.loads((project_root / case["evaluation"]["path"]).read_text()) for case in protocol["cases"]]
        return {"denominator": len(outcomes), "labels": labels,
                "records": [{"case_id": row["case_id"], "construction_status": row["construction_status"],
                             "assessment_status": row["assessment"]["status"]} for row in outcomes]}

    runner = hundred.create_runner(project_root=root, adapter=single, scenario_simulator=simulate,
                                   scorer=scorer, expected_compounds=3)
    return SimpleNamespace(root=root, run=run, manifest=root / manifest["path"], cases=cases,
                           runner=runner, calls=calls, score_calls=score_calls, simulate=simulate)


def prepare(batch, workers=2):
    return batch.runner.prepare(batch.manifest, batch.run, workers=workers)


def test_exact_hundred_distinct_identity_contract(batch):
    manifest = {"version": hundred.MANIFEST_VERSION, "cases": [
        {"case_id": f"case_{i}", "compound_id": f"drug:{i}", "identity_smiles": "C" * (i + 1),
         "input": {"path": "not_opened.json", "sha256": "a" * 64},
         "evaluation_scope": "aggregated_human_IV_PK_parameters"} for i in range(100)]}
    batch.runner.expected_compounds = 100
    assert len(batch.runner._manifest_cases(manifest)) == 100
    with pytest.raises(ValueError, match="exactly 100"):
        batch.runner._manifest_cases({**manifest, "cases": manifest["cases"][:-1]})
    duplicate = deepcopy(manifest)
    duplicate["cases"][-1]["compound_id"] = "drug:0"
    with pytest.raises(ValueError, match="distinct"):
        batch.runner._manifest_cases(duplicate)
    duplicate = deepcopy(manifest)
    duplicate["cases"][0]["identity_smiles"] = "CCO"
    duplicate["cases"][1]["identity_smiles"] = "OCC"
    with pytest.raises(ValueError, match="normalized chemical"):
        batch.runner._manifest_cases(duplicate)


def test_prepare_preserves_existing_curation_and_rejects_execution_overwrite(batch):
    source = put(batch.root, str(batch.run.relative_to(batch.root) / "sources/prepared.json"), {"preserve": True})
    prepare(batch)
    assert sha256(batch.root / source["path"]) == source["sha256"]
    with pytest.raises(FileExistsError):
        prepare(batch)


def test_prepare_and_execution_never_decode_evaluation_before_global_lock(batch, monkeypatch):
    original = batch.runner.read
    evaluation_paths = {str((batch.root / row["evaluation"]["path"]).resolve()) for row in batch.cases}
    def read(path):
        assert str((batch.root / path).resolve()) not in evaluation_paths
        return original(path)
    monkeypatch.setattr(batch.runner, "read", read)
    assert prepare(batch)["planned_scenarios"] == 9
    with pytest.raises(ValueError, match="must be locked"):
        batch.runner.report(batch.run)
    assert not batch.score_calls
    assert batch.runner.execute(batch.run)["status"] == "predictions_locked"
    assert not batch.score_calls and len(batch.calls) == 9
    result = batch.runner.report(batch.run)
    assert result["denominator"] == 3 and batch.score_calls == [3]
    assert result["total_registered_compounds"] == 3
    assert result["one_hundred_clinical_curves_claimed"] is False
    assert (batch.run / "results.csv").is_file()


@pytest.mark.parametrize("changed", ["source", "paper", "code"])
def test_source_code_and_publication_changes_refuse_resume(batch, changed):
    prepare(batch)
    target = {"source": batch.root / "sources/static.json", "paper": batch.root / "results/manuscript/main.tex",
              "code": batch.root / "src/pbpk_kb/blueprint_policy/purpose_design.py"}[changed]
    target.write_text(target.read_text() + "\nchanged\n")
    with pytest.raises(ValueError, match="changed"):
        batch.runner.execute(batch.run)
    assert batch.calls == []


def test_failure_is_locked_in_the_full_denominator_and_nominal_is_identical(batch):
    def failing(protocol, scenario, folder, guard):
        if protocol["input"]["request"]["compound_id"] == "synthetic:1" and scenario["scenario_id"] == "baseline":
            raise ValueError("Synthetic unestimable nominal")
        return batch.simulate(protocol, scenario, folder, guard)
    batch.runner.simulate = failing
    prepare(batch)
    batch.runner.execute(batch.run)
    protocol, state = batch.runner.load(batch.run, locked=True)
    outcomes = [batch.runner.checked(value) for value in state["outcomes"].values()]
    assert len(outcomes) == 3
    failed = next(row for row in outcomes if row["case_id"] == "drug_1")
    assert failed["construction_status"] == "failed"
    assert failed["assessment"]["status"] == "needs_evidence"
    for row in outcomes:
        baseline = next(prediction for prediction in row["predictions"] if prediction["scenario_id"] == "baseline")
        assert row["fixedauto_baseline"]["prediction_sha256"] == hundred.digest(baseline)
    assert batch.runner.report(batch.run)["denominator"] == 3


def test_complete_case_cache_and_report_are_idempotent_and_artifacts_bound(batch):
    prepare(batch)
    batch.runner.execute(batch.run)
    result = batch.runner.report(batch.run)
    assert batch.runner.execute(batch.run)["reused"] is True
    assert batch.runner.report(batch.run) == result
    assert len(batch.calls) == 9 and batch.score_calls == [3]
    artifact = batch.run / "work/drug_0/baseline/native_audit.txt"
    artifact.write_text("tampered")
    with pytest.raises(ValueError, match="Source changed"):
        batch.runner.report(batch.run)


def test_inline_state_cannot_replace_locked_predictions(batch):
    prepare(batch)
    batch.runner.execute(batch.run)
    state = batch.runner.read(batch.run / "state.json")
    outcome = batch.runner.checked(state["outcomes"]["drug_0"])
    outcome["predictions"][0]["endpoints"]["auc"]["value"] = 999.
    save(batch.root / state["outcomes"]["drug_0"]["path"], outcome)
    # Even a refreshed outer pointer cannot validate altered inline predictions.
    state["outcomes"]["drug_0"] = batch.runner.ref(batch.run / "work/drug_0/outcome.json")
    save(batch.run / "state.json", state)
    with pytest.raises(ValueError, match="Inline prediction checksum"):
        batch.runner.report(batch.run)


def test_interrupted_native_is_retained_and_never_retried(batch):
    prepare(batch, workers=1)
    partial = batch.run / "work/drug_0/baseline"
    partial.mkdir(parents=True)
    (partial / "started.json").write_text('{"interrupted": true}')
    (partial / "partial_native.txt").write_text("retain this diagnostic")
    batch.runner.execute(batch.run)
    assert ("synthetic:0", "baseline") not in batch.calls
    assert len(batch.calls) == 8
    _, state = batch.runner.load(batch.run, locked=True)
    outcome = batch.runner.checked(state["outcomes"]["drug_0"])
    assert outcome["predictions"][0]["failure_kind"] == "interrupted_native_scenario"
    assert (partial / "partial_native.txt").read_text() == "retain this diagnostic"


def test_three_consecutive_infrastructure_errors_pause_new_submissions_and_resume_cache(batch):
    attempted = []
    def infra(protocol, scenario, folder, guard):
        attempted.append(scenario["scenario_id"])
        raise RuntimeError("synthetic native service unavailable")
    batch.runner.simulate = infra
    prepare(batch, workers=1)
    result = batch.runner.execute(batch.run)
    assert result["status"] == "paused" and len(attempted) == 3
    assert len(batch.runner.read(batch.run / "state.json")["outcomes"]) == 1
    batch.runner.simulate = batch.simulate
    result = batch.runner.resume(batch.run)
    assert result["status"] == "predictions_locked"
    assert len(batch.calls) == 6
    assert batch.runner.report(batch.run)["denominator"] == 3


def test_pause_blocks_new_native_and_disk_guard_stops_submission(batch, monkeypatch):
    prepare(batch)
    batch.runner.pause(batch.run)
    assert batch.runner.execute(batch.run)["status"] == "paused"
    assert not batch.calls
    (batch.run / "PAUSE").unlink()
    monkeypatch.setattr(batch.runner, "directory_size", lambda _: hundred.LIMIT_BYTES + 1)
    with pytest.raises(hundred.Paused, match="10 GiB"):
        batch.runner.execute(batch.run)
    assert not batch.calls
    assert batch.runner.read(batch.run / "state.json")["status"] == "paused"


def test_background_restores_stale_pid_and_accepts_only_this_script_command(batch, monkeypatch):
    import psutil
    prepare(batch)
    processes = {}
    created = []
    class Process:
        def __init__(self, pid):
            if pid not in processes:
                raise psutil.NoSuchProcess(pid)
            self.pid = pid
        def create_time(self): return processes[self.pid]["created"]
        def is_running(self): return True
        def status(self): return psutil.STATUS_RUNNING
        def uids(self): return SimpleNamespace(real=hundred.os.getuid())
        def cmdline(self): return processes[self.pid]["argv"]
        def cwd(self): return str(batch.root)
    def popen(argv, **kwargs):
        assert kwargs["cwd"] == batch.root and kwargs["start_new_session"] is True
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert Path(kwargs["env"]["TMPDIR"]).is_relative_to(batch.root)
        assert "--report-after-run" in argv
        pid = 99001 + len(created)
        processes[pid] = {"created": float(pid), "argv": argv}
        created.append(argv)
        return SimpleNamespace(pid=pid)
    monkeypatch.setattr(psutil, "Process", Process)
    monkeypatch.setattr(hundred.subprocess, "Popen", popen)
    first = batch.runner.background(batch.run)
    assert first["status"] == "started"
    assert batch.runner.background(batch.run)["status"] == "already_running"
    assert len(created) == 1
    del processes[first["pid"]]
    assert batch.runner.background(batch.run)["status"] == "started"
    assert len(created) == 2
    processes[99002]["argv"] = ["unrelated-process"]
    with pytest.raises(ValueError, match="command"):
        batch.runner.background(batch.run)
    assert len(created) == 2


def test_wrong_project_reference_and_eval_as_input_are_rejected(batch):
    manifest = batch.runner.read(batch.manifest)
    manifest["cases"][0]["input"]["path"] = "../outside.json"
    save(batch.manifest, manifest)
    with pytest.raises(ValueError, match="inside the project"):
        prepare(batch)
    manifest["cases"][0]["input"] = batch.cases[0]["input"]
    manifest["cases"][0]["evaluation"] = batch.cases[0]["input"]
    save(batch.manifest, manifest)
    # Input-reference equality is a forbidden evaluation dependency, not permission
    # to decode an outcome before the prediction lock.
    with pytest.raises(ValueError, match="forbidden"):
        prepare(batch)


def test_foreign_case_evaluation_cannot_be_decoded_as_a_ledger(batch, monkeypatch):
    manifest = batch.runner.read(batch.manifest)
    input_path = batch.root / manifest["cases"][0]["input"]["path"]
    document = batch.runner.read(input_path)
    held_out = manifest["cases"][1]["evaluation"]
    document["request"]["evidence_ledger"] = {**held_out, "location": "not visible"}
    save(input_path, document)
    manifest["cases"][0]["input"] = batch.runner.ref(input_path)
    save(batch.manifest, manifest)
    original = single.read
    def never_decode(path):
        assert (batch.root / path).resolve() != (batch.root / held_out["path"]).resolve()
        return original(path)
    monkeypatch.setattr(single, "read", never_decode)
    with pytest.raises(ValueError, match="forbidden"):
        prepare(batch)


def test_blocked_construction_remains_a_locked_case_in_denominator(batch):
    manifest = batch.runner.read(batch.manifest)
    case = manifest["cases"][0]
    document = batch.runner.checked(case["input"])
    document["request"]["evidence"] = [row for row in document["request"]["evidence"] if row["parameter"] != "molecular_weight"]
    document["request"]["evidence_ledger"] = {**put(batch.root, "inputs/drug_0/blocked_ledger.json",
        {"evidence": document["request"]["evidence"]}), "location": "evidence"}
    save(batch.root / case["input"]["path"], document)
    case["input"] = batch.runner.ref(case["input"]["path"])
    save(batch.manifest, manifest)
    assert prepare(batch)["planned_scenarios"] == 6
    batch.runner.execute(batch.run)
    _, state = batch.runner.load(batch.run, locked=True)
    outcome = batch.runner.checked(state["outcomes"]["drug_0"])
    assert outcome["construction_status"] == "blocked" and outcome["predictions"] == []
    assert outcome["assessment"]["status"] == "needs_evidence"
    assert batch.runner.report(batch.run)["denominator"] == 3


def test_real_scorer_accepts_locked_contract_and_unicode_prediction_hashes(batch):
    from pbpk_kb.blueprint_policy import hundred_evaluation
    manifest = batch.runner.read(batch.manifest)
    for case in manifest["cases"]:
        evaluation = {key: case[key] for key in ("compound_id", "identity_smiles", "evaluation_scope")}
        evaluation.update(weight_kg=70., clinical_curve=None, endpoints={
            "clearance": {"value": 10., "unit": "l/h"}, "vss": {"value": 70., "unit": "l"},
            "terminal_half_life": {"value": 3., "unit": "h"}, "mrt": {"value": 4., "unit": "h"}})
        path = batch.root / case["evaluation"]["path"]
        save(path, evaluation)
        case["evaluation"] = batch.runner.ref(path)
    save(batch.manifest, manifest)
    def unicode_prediction(protocol, scenario, folder, guard):
        result = batch.simulate(protocol, scenario, folder, guard)
        result["observer_note"] = "合成观察位置"
        return result
    batch.runner.simulate = unicode_prediction
    batch.runner.scorer = hundred_evaluation.score_hundred
    prepare(batch)
    batch.runner.execute(batch.run)
    result = batch.runner.report(batch.run)
    assert result["construction"]["total"] == 3
    assert result["clinical_curve_validation_n"] == 0
    assert len(result["records"]) == 33
    assert result["same_nominal_predictions_for_both_methods"] is True


def test_mixed_calibration_regimes_cannot_enter_zero_curve_batch(batch):
    manifest = batch.runner.read(batch.manifest)
    case = manifest["cases"][0]
    document = batch.runner.checked(case["input"])
    document["calibration_mapping"] = put(batch.root, "unread_mapping.json", {"do_not_use": True})
    save(batch.root / case["input"]["path"], document)
    case["input"] = batch.runner.ref(case["input"]["path"])
    save(batch.manifest, manifest)
    with pytest.raises(ValueError, match="zero-curve uncalibrated"):
        prepare(batch)


def test_background_worker_cli_automatically_reports_without_repeating_predictions(batch, capsys):
    prepare(batch)
    batch.runner.main(["run", "--run", str(batch.run), "--report-after-run"])
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"
    assert batch.score_calls == [3] and len(batch.calls) == 9
    batch.runner.main(["run", "--run", str(batch.run), "--report-after-run"])
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    assert batch.score_calls == [3] and len(batch.calls) == 9
