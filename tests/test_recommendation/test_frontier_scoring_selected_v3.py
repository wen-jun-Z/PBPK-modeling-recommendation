"""CPU checks for conditional selection, immutable pilot reuse and full handoff."""
import copy
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
v = importlib.import_module("run_frontier_scoring_selected_v3")


@pytest.fixture(autouse=True)
def restore_modules():
    old = {key: getattr(v.base, key) for key in ("SYSTEM", "ASSESS", "RUN", "prepare", "GPUJudge", "download_primary")}
    parents = (v.parent.ASSESS, v.parent.RUN, v.utilities.ASSESS)
    yield
    for key, value in old.items(): setattr(v.base, key, value)
    v.parent.ASSESS, v.parent.RUN, v.utilities.ASSESS = parents


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def reports():
    metric = {"complete": True, "controlled_semantic_accuracy": 1.,
        "controlled_citation_support_accuracy": 1., "citation_numbers_valid_fraction": 1., "raw_format_valid_fraction": 1.}
    report = {"complete": True, "admission_by_role": {role: copy.deepcopy(metric) for role in v.CONTROL_ROLES},
        "by_role": {"all": {"rows": [{"schema_valid": True, "citation_numbers_valid": True} for _ in range(733)]}},
        "final_validation_by_role": {"prospective_source_coverage": {"final_valid_fraction": 1.}},
        "coverage_sequence": {"complete": True, "any_full_window_above10percent": False}}
    return {"phi": copy.deepcopy(report), "qwen": copy.deepcopy(report)}


def test_qualified_qwen_can_be_selected_while_phi_failed_without_relabelling():
    values = reports()
    values["phi"]["admission_by_role"][v.CONTROL_ROLES[1]]["controlled_semantic_accuracy"] = 56/64
    result = v.admission_from_reports(values)
    assert result["accepted"] and not result["parent_phi_numeric_admission"]
    assert result["primary"] == "Qwen/Qwen3.6-27B" and result["secondary"] == "microsoft/phi-4"
    assert result["control_groups"][v.CONTROL_ROLES[1]]["phi"]["controlled_semantic_accuracy"] == 56/64
    assert result["thresholds"] == v.utilities.GATE and result["semantic_gold_accuracy"] is None


@pytest.mark.parametrize("role", v.CONTROL_ROLES)
def test_any_qwen_control_failure_blocks_selection(role):
    values = reports()
    values["qwen"]["admission_by_role"][role]["controlled_semantic_accuracy"] = .89
    assert not v.admission_from_reports(values)["accepted"]


def test_original_relative_gate_is_not_waived_for_new_primary():
    values = reports()
    values["qwen"]["admission_by_role"][v.CONTROL_ROLES[0]]["controlled_semantic_accuracy"] = .95
    assert not v.admission_from_reports(values)["accepted"]


@pytest.mark.parametrize("change", ["coverage", "window", "incomplete", "format", "citation"])
def test_same_engineering_gates_apply_to_qwen(change):
    values = reports()
    if change == "coverage": values["qwen"]["final_validation_by_role"]["prospective_source_coverage"]["final_valid_fraction"] = .979
    elif change == "window": values["qwen"]["coverage_sequence"]["any_full_window_above10percent"] = True
    elif change == "incomplete": values["phi"]["complete"] = False
    else:
        key = "schema_valid" if change == "format" else "citation_numbers_valid"
        for row in values["qwen"]["by_role"]["all"]["rows"][:20]: row[key] = False
    assert not v.admission_from_reports(values)["accepted"]


@pytest.fixture
def paths(tmp_path, monkeypatch):
    assess, old, run = tmp_path / "selected", tmp_path / "parent", tmp_path / "run"
    assess.mkdir(); old.mkdir(); run.mkdir()
    for name, path in (("ROOT", tmp_path), ("ASSESS", assess), ("PARENT_ASSESS", old), ("RUN", run)):
        monkeypatch.setattr(v, name, path)
    return assess, old, run


def test_only_individual_validation_files_are_linked_and_old_admission_survives(paths):
    assess, old, _ = paths
    for name in v.PILOT_FILES: put(old / "validation" / name, {"fixture": name})
    put(old / "source_review_plan.json", {})
    put(old / "validation/admission.json", {"accepted": False})
    v.link_validation()
    assert not (assess / "validation").is_symlink()
    assert (assess / "validation/phi_records.jsonl").is_symlink()
    assert not (assess / "validation/admission.json").exists()
    put(assess / "validation/admission.json", {"accepted": True})
    assert v.base.read(old / "validation/admission.json") == {"accepted": False}
    assert not (assess / "judgments.jsonl").exists()
    v.link_validation()


def test_validation_directory_link_is_refused(paths):
    assess, old, _ = paths
    (old / "validation").mkdir()
    (assess / "validation").symlink_to(old / "validation", target_is_directory=True)
    with pytest.raises(ValueError, match="directory symlink"): v.link_validation()


def test_old_labels_symlink_is_refused(paths):
    assess, old, _ = paths
    for name in v.PILOT_FILES: put(old / "validation" / name, {})
    put(old / "source_review_plan.json", {})
    put(old / "judgments.jsonl", {})
    (assess / "judgments.jsonl").symlink_to(old / "judgments.jsonl")
    with pytest.raises(ValueError, match="Writable selected-run"): v.link_validation()


@pytest.fixture
def pilot(paths, monkeypatch):
    assess, old, run = paths
    weights = assess / "weights.json"
    put(weights, {"repo": "fixture-qwen", "revision": "pinned"})
    monkeypatch.setattr(v, "QWEN_WEIGHTS", weights)
    folder = assess / "validation"
    protocol = {"model_repo": "fixture-qwen", "model_revision": "pinned", "judge_protocol_sha256": "parent-hash",
        "validation_sha256": "case-hash", "maximum_batch_size": 16, "memory_gib": 36, "raw_max_tokens": 96,
        "repair_max_tokens": 160, "control_repairs": 0, "noncontrol_repairs": 1, "raw_journal_before_repairs": True}
    put(folder / "qwen_protocol.json", protocol)
    cases = [{"case_id": "control", "kind": "synthetic_control"}, {"case_id": "source", "kind": "real_source_pair"}]
    rows = [{**c, "model_repo": "fixture-qwen", "model_revision": "pinned", "validation_sha256": "case-hash",
        "pilot_protocol_sha256": v.base.file_sha(folder / "qwen_protocol.json"), "output": "raw",
        "final_output": "raw", "attempts": [{"output": "raw"}]} for c in cases]
    v.base.append_rows(folder / "qwen_records.jsonl", rows)
    v.base.append_rows(folder / "qwen_raw_generation_journal.jsonl", rows)
    return folder, cases, rows


def test_reused_pilot_exact_cohort_and_raw_provenance_verified_without_writes(pilot):
    folder, cases, rows = pilot
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert v.verify_pilot_records("qwen", cases, "parent-hash", "case-hash") == rows
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


@pytest.mark.parametrize("change", ["missing", "duplicate", "raw_revision", "raw_protocol", "attempt", "truncated", "generation"])
def test_reuse_rejects_partial_mixed_or_different_generation(pilot, change):
    folder, cases, rows = pilot
    path = folder / "qwen_records.jsonl"
    if change == "missing": path.write_text(json.dumps(rows[0])+"\n")
    elif change == "duplicate": v.base.append_rows(path, [rows[0]])
    elif change.startswith("raw_"):
        rows[0]["model_revision" if change == "raw_revision" else "pilot_protocol_sha256"] = "wrong"
        path = folder / "qwen_raw_generation_journal.jsonl"
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    elif change == "attempt":
        rows[0]["attempts"].append({"output": "raw"})
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    elif change == "truncated":
        with path.open("a") as handle: handle.write('{"unfinished":')
    else:
        path = folder / "qwen_protocol.json"
        protocol = v.base.read(path); protocol["raw_max_tokens"] = 160; put(path, protocol)
    before = path.read_bytes()
    with pytest.raises(ValueError): v.verify_pilot_records("qwen", cases, "parent-hash", "case-hash")
    assert path.read_bytes() == before


def test_parent_running_blocks_before_protocol_creation(paths, monkeypatch):
    assess, old, run = paths
    monkeypatch.setattr(v.base, "alive", lambda value: value.get("fixture_alive", False))
    monkeypatch.setattr(v, "PARENT_RUN", old)
    put(old / "pipeline_pid.json", {"fixture_alive": True})
    with pytest.raises(RuntimeError, match="still running"): v.prepare()
    assert not (assess / "judge_protocol.json").exists()


@pytest.mark.parametrize("judge,model,weights", [("primary", "qwen", "QWEN_WEIGHTS"), ("secondary", "phi", "PHI_WEIGHTS")])
def test_full_handoff_has_true_models_and_no_pilot_call(paths, monkeypatch, judge, model, weights):
    captured = {}
    monkeypatch.setattr(v, "check_gate", lambda: {"accepted": True})
    monkeypatch.setattr(v.parent, "backend", lambda name: captured.update(backend=name) or name)
    monkeypatch.setattr(v.utilities, "verify_weights", lambda path: captured.update(primary_weights=path))
    def invoke(args):
        captured["args"] = args
        v.base.download_primary()
    monkeypatch.setattr(v.base, "run_assessment", invoke)
    v.run_full(judge)
    assert captured["backend"] == model
    assert captured["primary_weights"] == v.QWEN_WEIGHTS
    assert captured["args"].secondary_weights == v.PHI_WEIGHTS
    assert captured["args"].batch_size == 16 and captured["args"].memory_gib == 36
    assert v.base.prepare is v.prepare


def test_rejected_gate_never_constructs_gpu_backend(paths, monkeypatch):
    monkeypatch.setattr(v, "check_gate", lambda: {"accepted": False})
    monkeypatch.setattr(v.parent, "backend", lambda _: pytest.fail("Blocked primary cannot load GPU"))
    with pytest.raises(RuntimeError, match="remains blocked"): v.run_full("primary")


def test_quality_stopped_new_run_cannot_restart_identically(paths):
    assess, old, run = paths
    put(run / "status.json", {"guard_reason": "PrimaryInvalidFractionAbove10Percent"})
    with pytest.raises(RuntimeError, match="not an identical restart"): v.start()


def test_queue_only_two_full_stages_using_correct_interpreters():
    assert v.stages() == [("primary", v.MODERN, ["run", "--judge", "primary"]),
                          ("secondary", v.APP, ["run", "--judge", "secondary"])]
    assert v.command(v.MODERN, "run", "--judge", "primary")[1].endswith("run_frontier_scoring_selected_v3.py")


def test_review_additionally_binds_parent_and_real_selected_roles(paths, monkeypatch):
    assess, old, _ = paths
    put(old / "judge_protocol.json", {"parent": True})
    monkeypatch.setattr(v.parent, "verify_source_review", lambda: (True, "review-hash"))
    value = {"parent_protocol_sha256": v.base.file_sha(old / "judge_protocol.json"), "primary": "qwen", "secondary": "phi"}
    put(assess / "source_review.json", value)
    assert v.verify_source_review() == (True, "review-hash")
    value["primary"] = "phi"; put(assess / "source_review.json", value)
    with pytest.raises(ValueError, match="selected judge roles"): v.verify_source_review()


def test_frozen_parent_code_and_amendment_remain_original():
    assert v.base.file_sha(v.ROOT / "scripts/run_frontier_scoring_v3.py") == "f389c9093640325320f5ed7306fd0151f87f88506878bbb5dda1e45d909cc2f2"
    assert v.verify_amendment()["execution_policy"]["ranking_scores_used_in_judge_selection"] is False


def test_protocol_clone_preserves_generation_and_freezes_true_new_identity(paths, monkeypatch):
    assess, old, _ = paths
    put(assess / "phi.json", {"repo": "phi", "revision": "phi-pinned"})
    put(assess / "qwen.json", {"repo": "qwen", "revision": "qwen-pinned"})
    monkeypatch.setattr(v, "PHI_WEIGHTS", assess / "phi.json")
    monkeypatch.setattr(v, "QWEN_WEIGHTS", assess / "qwen.json")
    generation = {"max_new_tokens": 96, "retry_max_new_tokens": 160, "max_attempts": 2,
                  "constrained_decoding": {"same": "evidence-first"}}
    old_protocol = {"implementation_sha256": {}, "system_prompt": v.SYSTEM,
        "system_prompt_sha256": v.base.sha(v.SYSTEM.encode()), "generation": generation,
        "execution_environments": {"qwen": "modern", "phi": "app"},
        "primary_repo": "phi", "primary_revision": "phi-pinned", "secondary_repo": "qwen", "secondary_revision": "qwen-pinned",
        "primary_weights_manifest_sha256": v.base.file_sha(v.PHI_WEIGHTS),
        "secondary_weights_manifest_sha256": v.base.file_sha(v.QWEN_WEIGHTS),
        "input_sha256": {"assessment/sources.jsonl": "unchanged-source"}, "scope_guard": {"version": "unchanged"},
        "pilot_labels_reused": False}
    put(old / "judge_protocol.json", old_protocol)
    for name in v.PILOT_FILES: put(old / "validation" / name, {"fixture": name})
    (old / "validation/cases.jsonl").write_text('{"case_id":"one"}\n')
    put(old / "source_review_plan.json", {})
    put(assess / "sources.jsonl", {"fixture": "source"})
    amendment = {"parent_protocol_sha256": v.base.file_sha(old / "judge_protocol.json"),
        "parent_validation_manifest_sha256": v.base.file_sha(old / "validation/manifest.json"), "selection_timing": "declared before Qwen"}
    monkeypatch.setattr(v, "assert_other_runs_idle", lambda: None)
    monkeypatch.setattr(v, "verify_amendment", lambda: amendment)
    monkeypatch.setattr(v.utilities, "execution_environments", lambda: old_protocol["execution_environments"])
    decoder = importlib.import_module("frontier_evidence_first_decoding")
    monkeypatch.setattr(decoder, "decoding_protocol", lambda: generation["constrained_decoding"])
    monkeypatch.setattr(v.utilities, "prepare_materials", lambda: ([{}], [{}, {}],
        {"one": {"packet_sha256": "p1", "refs": []}, "two": {"packet_sha256": "p2", "refs": [1]}}, {}))
    validator = importlib.import_module("frontier_judge_validation_v3")
    monkeypatch.setattr(validator, "verify_validation_v3", lambda *args: {})
    monkeypatch.setattr(v.parent, "verify_review_plan", lambda cases: {})
    monkeypatch.setattr(v, "verify_pilot_records", lambda *args: [])
    *_, protocol = v.prepare()
    assert protocol["primary_repo"] == "qwen" and protocol["secondary_repo"] == "phi"
    assert protocol["primary_weights_manifest_sha256"] == v.base.file_sha(v.QWEN_WEIGHTS)
    assert protocol["generation"] == generation and protocol["scope_guard"] == old_protocol["scope_guard"]
    assert protocol["input_sha256"] == old_protocol["input_sha256"]
    assert not protocol["full_labels_reused"] and not protocol["pilot_labels_reused"]
    assert protocol["parent_pilot_outputs_reused_for_validation"]
    assert protocol["implementation_sha256"][Path(v.__file__).name] == v.base.file_sha(Path(v.__file__))
    assert v.prepare()[3] == protocol
    assert v.base.read(old / "judge_protocol.json") == old_protocol


@pytest.fixture
def scheduler(paths, monkeypatch):
    assess, old, run = paths
    monkeypatch.setattr(v, "assert_other_runs_idle", lambda: None)
    monkeypatch.setattr(v, "check_gate", lambda require_review=True: {"numeric_accepted": True, "accepted": True})
    monkeypatch.setattr(v, "publish_summary", lambda: None)
    monkeypatch.setattr(v.signal, "signal", lambda *args: None)
    monkeypatch.setattr(v.base, "identity", lambda: {"pid": 100})
    monkeypatch.setattr(v.base, "stop_child", lambda child: None)
    put(assess / "source_review.json", {})
    put(assess / "judge_protocol.json", {})
    monkeypatch.setattr(v, "verify_source_review", lambda: (True, "review-sha"))
    return assess, run


def test_pipeline_failed_worker_retains_actual_final_count_and_does_not_run_secondary(scheduler, monkeypatch):
    assess, run = scheduler
    calls = []
    class Child:
        pid = 200
        returncode = 1
        def __init__(self, command, **kwargs):
            calls.append(command)
            put(run / "status.json", {"pid": self.pid, "state": "failed", "committed_pairs": 4680,
                "guard_reason": "PrimaryInvalidFractionAbove10Percent"})
        def poll(self): return self.returncode
    monkeypatch.setattr(v.subprocess, "Popen", Child)
    v.pipeline()
    status = v.base.read(run / "pipeline_status.json")
    assert status["state"] == "failed" and status["committed_pairs"] == 4680
    assert status["guard_reason"] == "PrimaryInvalidFractionAbove10Percent"
    assert len(calls) == 1 and calls[0][-1] == "primary"


def test_complete_pipeline_runs_full_stages_strict_metrics_and_true_agreement_identity(scheduler, monkeypatch):
    assess, run = scheduler
    calls, metrics = [], []
    class Child:
        returncode = 0
        def __init__(self, command, **kwargs):
            calls.append(command); self.pid = 200+len(calls)
            put(run / "status.json", {"pid": self.pid, "state": "complete", "judge": command[-1],
                "committed_pairs": 216160 if command[-1] == "primary" else 10771})
        def poll(self): return self.returncode
    monkeypatch.setattr(v.subprocess, "Popen", Child)
    evaluator = importlib.import_module("evaluate_frontier_automatic")
    monkeypatch.setattr(evaluator, "evaluate_run", lambda out, target, **kwargs: metrics.append((target, kwargs)) or {"state": "complete"})
    monkeypatch.setattr(v.base, "judge_agreement", lambda: {"paired": 10771})
    v.pipeline()
    assert [c[-1] for c in calls] == ["primary", "secondary"]
    assert [c[0] for c in calls] == [str(v.MODERN), str(v.APP)]
    assert len(metrics) == 2 and all(m == (assess, {"require_complete": True}) for m in metrics)
    assert v.base.read(run / "pipeline_status.json")["state"] == "complete"
    agreement = v.base.read(assess / "judge_agreement.json")
    assert agreement["primary_repo"] == "Qwen/Qwen3.6-27B" and agreement["secondary_repo"] == "microsoft/phi-4"
    assert agreement["source_review_sha256"] == "review-sha"


def test_numeric_failure_does_not_enter_source_wait_or_launch(scheduler, monkeypatch):
    assess, run = scheduler
    monkeypatch.setattr(v, "check_gate", lambda require_review=True: {"numeric_accepted": False, "accepted": False})
    monkeypatch.setattr(v.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("No child after failed gate"))
    v.pipeline()
    assert v.base.read(run / "pipeline_status.json")["state"] == "stopped_on_validation"


def test_secondary_summary_reports_own_denominator(paths):
    assess, old, run = paths
    folder = v.ROOT / "experiments/model_recommendation/results_summary"
    folder.mkdir(parents=True)
    (folder / "README.md").write_text("Existing results\n")
    put(run / "status.json", {"judge": "secondary", "committed_pairs": 10771, "expected_pairs": 10771})
    v.publish_summary()
    text = (folder / "README.md").read_text()
    assert "5% 复判已提交：10,771/10,771" in text and "10,771/216,160" not in text
