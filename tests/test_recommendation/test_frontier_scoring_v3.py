"""CPU checks for v3 provenance, repair limits, admission and lifecycle."""
import copy
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
v3 = importlib.import_module("run_frontier_scoring_v3")


@pytest.fixture(autouse=True)
def restore_base():
    old = {key: getattr(v3.base, key) for key in ("SYSTEM", "ASSESS", "RUN", "prepare", "GPUJudge", "download_primary")}
    assess = v3.utilities.ASSESS
    yield
    for key, value in old.items(): setattr(v3.base, key, value)
    v3.utilities.ASSESS = assess


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def reports_fixture():
    metric = {"complete": True, "controlled_semantic_accuracy": 1.,
        "controlled_citation_support_accuracy": 1., "citation_numbers_valid_fraction": 1., "raw_format_valid_fraction": 1.}
    report = {"complete": True, "admission_by_role": {role: copy.deepcopy(metric) for role in v3.CONTROL_ROLES},
        "by_role": {"all": {"rows": [{"schema_valid": True, "citation_numbers_valid": True} for _ in range(733)]}},
        "final_validation_by_role": {"prospective_source_coverage": {"final_valid_fraction": 1.}},
        "coverage_sequence": {"complete": True, "any_full_window_above10percent": False}}
    return {"phi": copy.deepcopy(report), "qwen": copy.deepcopy(report)}


def test_original_thresholds_and_three_control_groups_remain_separate():
    reports = reports_fixture()
    result = v3.admission_from_reports(reports)
    assert result["accepted"] is True
    assert result["thresholds"] == v3.utilities.GATE
    assert set(result["control_groups"]) == set(v3.CONTROL_ROLES)
    assert result["semantic_gold_accuracy"] is None and not result["real_source_entailment_proven"]


@pytest.mark.parametrize("role", v3.CONTROL_ROLES)
def test_each_control_role_must_independently_pass(role):
    reports = reports_fixture()
    reports["phi"]["admission_by_role"][role]["controlled_semantic_accuracy"] = .89
    assert not v3.admission_from_reports(reports)["accepted"]


@pytest.mark.parametrize("change", ["coverage", "window", "incomplete", "format", "citation"])
def test_raw_format_and_final_repaired_coverage_both_required(change):
    reports = reports_fixture()
    if change == "coverage": reports["phi"]["final_validation_by_role"]["prospective_source_coverage"]["final_valid_fraction"] = .979
    elif change == "window": reports["phi"]["coverage_sequence"]["any_full_window_above10percent"] = True
    elif change == "incomplete": reports["qwen"]["complete"] = False
    else:
        key = "schema_valid" if change == "format" else "citation_numbers_valid"
        for row in reports["phi"]["by_role"]["all"]["rows"][:20]: row[key] = False
    assert not v3.admission_from_reports(reports)["accepted"]


@pytest.mark.parametrize("state", ["worker_guard", "stopped_on_validation", "stopped_on_source_review"])
def test_quality_failure_forbids_identical_restart(tmp_path, monkeypatch, state):
    monkeypatch.setattr(v3, "RUN", tmp_path)
    if state == "worker_guard": put(tmp_path / "status.json", {"guard_reason": "PrimaryInvalidFractionAbove10Percent"})
    else: put(tmp_path / "pipeline_status.json", {"state": state})
    with pytest.raises(RuntimeError, match="not an identical restart"):
        v3.ensure_not_quality_stopped()


@pytest.mark.parametrize("model", ["phi", "qwen"])
def test_evidence_first_decoder_wraps_both_actual_backends(monkeypatch, model):
    decoder = importlib.import_module("frontier_evidence_first_decoding")
    wrapped = []
    class Parent:
        def __init__(self, *args, **kwargs): self.json_constraint = "original"
    monkeypatch.setattr(v3.utilities, "PhiJudge", Parent)
    monkeypatch.setattr(v3.utilities, "backend", lambda name: Parent)
    monkeypatch.setattr(decoder, "evidence_first_constraint", lambda value: wrapped.append(value) or "evidence-first")
    assert v3.backend(model)({}, 36).json_constraint == "evidence-first"
    assert wrapped == ["original"]


@pytest.fixture
def pilot_fixture(tmp_path, monkeypatch):
    assess, runtime = tmp_path / "assessment", tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(v3, "ASSESS", assess)
    monkeypatch.setattr(v3, "RUN", runtime)
    monkeypatch.setattr(v3, "prepare", lambda: None)
    put(assess / "judge_protocol.json", {"fixture": True})
    put(assess / "validation/manifest.json", {"fixture": True})
    cases = [{"case_id": "control", "kind": "synthetic_control", "packet": {}, "query": {}},
             {"case_id": "real", "kind": "real_source_pair", "packet": {}, "query": {}}]
    v3.base.append_rows(assess / "validation/cases.jsonl", cases)
    monkeypatch.setattr(v3.utilities, "verify_weights", lambda p: {"repo": "fixture", "revision": "fixed"})
    monkeypatch.setattr(v3, "validation_error", lambda output, case: "raw error" if output == "raw-invalid" else None)
    validator = importlib.import_module("frontier_judge_validation_v3")
    monkeypatch.setattr(validator, "evaluate_validation_v3", lambda records, cases: {"complete": len(records) == len(cases)})
    return assess, runtime


def test_pilot_journals_raw_before_single_real_repair_and_never_repairs_controls(pilot_fixture, monkeypatch):
    assess, runtime = pilot_fixture
    calls = []
    class Judge:
        def __init__(self, *args): pass
        def select_batch_size(self, tasks, maximum): return len(tasks)
        def generate(self, tasks, max_new_tokens=96, repairs=None):
            calls.append((len(tasks), max_new_tokens, repairs))
            if repairs:
                assert len(v3.base.load_committed(assess / "validation/phi_raw_generation_journal.jsonl")) == 2
                return ["repaired"], {"wall_seconds": 1.}
            return ["raw-invalid"] * len(tasks), {"wall_seconds": 1.}
    monkeypatch.setattr(v3, "backend", lambda model: Judge)
    v3.pilot("phi")
    records = v3.base.load_committed(assess / "validation/phi_records.jsonl")
    assert len(records[0]["attempts"]) == 1 and records[0]["final_status"] == "invalid_output"
    assert len(records[1]["attempts"]) == 2 and records[1]["final_output"] == "repaired"
    assert records[1]["output"] == "raw-invalid"
    assert calls[1][1] == 160 and calls[1][2][0]["error"] == "raw error"
    raw_bytes = (assess / "validation/phi_raw_generation_journal.jsonl").read_bytes()
    monkeypatch.setattr(v3, "backend", lambda model: pytest.fail("Completed pilots must not load GPU again"))
    v3.pilot("phi")
    assert (assess / "validation/phi_raw_generation_journal.jsonl").read_bytes() == raw_bytes


@pytest.fixture
def review_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, "ASSESS", tmp_path)
    put(tmp_path / "judge_protocol.json", {"fixture": 1})
    put(tmp_path / "validation/manifest.json", {"fixture": 2})
    for model in ("phi", "qwen"): (tmp_path / "validation" / f"{model}_records.jsonl").write_text('{}\n')
    cases = [{"case_id": "failure", "strata": {"validation_role": "production_failure_diagnostic"}},
             {"case_id": "sample", "strata": {"validation_role": "prospective_source_coverage"}}]
    v3.base.append_rows(tmp_path / "validation/cases.jsonl", cases)
    plan = {"case_ids": ["failure", "sample"], "case_count": 2,
            "validation_manifest_sha256": v3.base.file_sha(tmp_path / "validation/manifest.json")}
    put(tmp_path / "source_review_plan.json", plan)
    monkeypatch.setattr(v3, "REVIEW_PLAN_SHA", v3.base.file_sha(tmp_path / "source_review_plan.json"))
    review = {"version": "pbpk-v3-source-review-v1", "protocol_sha256": v3.base.file_sha(tmp_path / "judge_protocol.json"),
        "validation_manifest_sha256": v3.base.file_sha(tmp_path / "validation/manifest.json"),
        "pilot_record_sha256": {model: v3.base.file_sha(tmp_path / "validation" / f"{model}_records.jsonl") for model in ("phi", "qwen")},
        "accepted": True, "reviewer": "Codex evidence audit; not expert gold", "checked_case_ids": ["failure", "sample"],
        "findings": ["fixture"], "known_unknown_distribution": {"known": 1}, "scientific_gold_accuracy": None}
    put(tmp_path / "source_review.json", review)
    return tmp_path, review


@pytest.mark.parametrize("change", [None, "changed_output", "missing_failure", "false_gold", "wrong_protocol"])
def test_source_review_binds_exact_outputs_and_required_cases(review_fixture, change):
    path, review = review_fixture
    if change is None:
        assert v3.verify_source_review()[0] is True; return
    if change == "changed_output": (path / "validation/phi_records.jsonl").write_text('changed\n')
    elif change == "missing_failure": review["checked_case_ids"] = ["sample"]
    elif change == "false_gold": review["scientific_gold_accuracy"] = 1.
    else: review["protocol_sha256"] = "wrong"
    put(path / "source_review.json", review)
    with pytest.raises(ValueError): v3.verify_source_review()


def test_full_scoring_cannot_bypass_source_review(monkeypatch):
    monkeypatch.setattr(v3, "ensure_not_quality_stopped", lambda: None)
    monkeypatch.setattr(v3, "check_gate", lambda: {"accepted": False})
    monkeypatch.setattr(v3, "backend", lambda *args: pytest.fail("No GPU before admission and review"))
    with pytest.raises(RuntimeError, match="blocked"): v3.run_full("primary")


def test_pipeline_copies_last_worker_commit_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, "RUN", tmp_path)
    monkeypatch.setattr(v3, "prepare", lambda: None)
    monkeypatch.setattr(v3, "ensure_not_quality_stopped", lambda: None)
    monkeypatch.setattr(v3, "publish_summary", lambda: None)
    monkeypatch.setattr(v3.signal, "signal", lambda *args: None)
    monkeypatch.setattr(v3.base, "stop_child", lambda *args: None)
    class Child:
        pid, returncode = 80000, 1
        def __init__(self, *args, **kwargs):
            put(tmp_path / "pilot_phi_status.json", {"pid": self.pid, "state": "failed", "committed_pairs": 4680,
                "guard_reason": "PrimaryInvalidFractionAbove10Percent"})
        def poll(self): return self.returncode
    monkeypatch.setattr(v3.subprocess, "Popen", Child)
    v3.pipeline()
    state = v3.base.read(tmp_path / "pipeline_status.json")
    assert state["state"] == "failed" and state["committed_pairs"] == 4680
    assert state["guard_reason"] == "PrimaryInvalidFractionAbove10Percent"


def test_review_wait_is_bounded_and_does_not_claim_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, "ASSESS", tmp_path / "missing")
    monkeypatch.setattr(v3, "RUN", tmp_path)
    monkeypatch.setattr(v3, "publish_summary", lambda: None)
    times = iter([0, 1201])
    monkeypatch.setattr(v3.time, "monotonic", lambda: next(times))
    states = []
    assert v3.wait_for_source_review(lambda **kwargs: states.append(kwargs)) is False
    assert states[0]["stage"] == "waiting_for_source_review"
    assert states[-1]["state"] == "stopped_waiting_for_source_review"


def test_secondary_readme_uses_secondary_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(v3, "ROOT", tmp_path)
    monkeypatch.setattr(v3, "RUN", tmp_path / "runtime")
    monkeypatch.setattr(v3, "ASSESS", tmp_path / "assessment")
    v3.ASSESS.mkdir()
    folder = tmp_path / "experiments/model_recommendation/results_summary"
    folder.mkdir(parents=True)
    (folder / "README.md").write_text("Existing results\n")
    put(v3.RUN / "pipeline_status.json", {"stage": "secondary", "state": "running"})
    put(v3.RUN / "status.json", {"judge": "secondary", "committed_pairs": 123, "expected_pairs": 1000})
    v3.publish_summary()
    text = (folder / "README.md").read_text()
    assert "5% 复判已提交：123/1,000" in text
    assert "123/216,160" not in text


@pytest.mark.parametrize("name,model", [("primary", "phi"), ("secondary", "qwen")])
def test_full_handoff_preserves_scope_guard_and_correct_model(monkeypatch, name, model):
    monkeypatch.setattr(v3, "ensure_not_quality_stopped", lambda: None)
    monkeypatch.setattr(v3, "check_gate", lambda: {"accepted": True})
    chosen, calls = [], []
    monkeypatch.setattr(v3, "backend", lambda value: chosen.append(value) or value)
    monkeypatch.setattr(v3.base, "run_assessment", lambda args: calls.append(args))
    original_guard = v3.base.validate_explicit_scope
    v3.run_full(name)
    assert chosen == [model] and calls[0].secondary_weights == v3.QWEN_WEIGHTS
    assert calls[0].batch_size == 16 and calls[0].memory_gib == 36
    assert v3.base.validate_explicit_scope is original_guard
