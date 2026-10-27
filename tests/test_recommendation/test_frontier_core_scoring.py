"""CPU integration for cohort isolation, qualification handoff and safe resume."""
import importlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
m = importlib.import_module("run_frontier_core_scoring")


@pytest.fixture(autouse=True)
def restore_modules():
    modules = (m.r, m.r.shared, m.r.engine, m.base, m.r.utilities, m.qualification, m.qualification.fast)
    snapshots = [(module, dict(vars(module))) for module in modules]
    environment = dict(os.environ)
    yield
    for module, values in snapshots:
        for name, value in values.items():
            setattr(module, name, value)
    os.environ.clear(); os.environ.update(environment)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v) + "\n" for v in values))


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    out, run = tmp_path / "experiments/core", tmp_path / "runtime/deployment/frontier_core_145"
    assess, qa, qr = out / "assessment", tmp_path / "experiments/old/assessment", tmp_path / "runtime/deployment/frontier_scoring_reviewed_v8"
    for folder in (out, run, assess, qa, qr):
        folder.mkdir(parents=True)
    for name, value in (("ROOT", tmp_path), ("OUT", out), ("ASSESS", assess), ("RUN", run),
                        ("QUAL_ASSESS", qa), ("QUAL_RUN", qr)):
        monkeypatch.setattr(m, name, value)
    script = tmp_path / "scripts/runner.py"; script.parent.mkdir(); script.write_text("pass\n")
    monkeypatch.setattr(m, "SCRIPT", script)
    monkeypatch.setattr(m, "initialize", lambda: None)
    queries = [{"query_id": f"q{i}", "compound_id": f"d{i//5}", "task_type": f"t{i%5}",
                "text": f"Predict endpoint {i}"} for i in range(145)]
    candidates = [{"candidate_id": f"c{i}"} for i in range(280)]
    packets = {c["candidate_id"]: {"packet_sha256": str(i), "refs": [{"quote": "source"}], "body": {}}
               for i, c in enumerate(candidates)}
    protocol = {"primary_repo": "Qwen/Qwen3.6-27B", "secondary_repo": "microsoft/phi-4",
                "generation": {"thinking_budget": 1024}, "implementation_sha256": {"unchanged.py": "frozen"},
                "scope_guard": {"version": "v8"}}
    put(qa / "judge_protocol.json", protocol)
    for name in ("sources.jsonl", "source_supplements/sources.jsonl", "measured_outcomes.json"):
        put(qa / name, {"source": name})
    rows(out / "inputs/queries.jsonl", queries); rows(out / "inputs/candidates.jsonl", candidates)
    methods = [f"same-ranking-{i}" for i in range(12)]
    provenance = {}
    for method in methods:
        folder = out / "results" / method
        for name in ("scores.npy", "query_seconds.npy", "index.json", "status.json"):
            put(folder / name, {"frozen": name})
        path = folder / "scores.npy"
        provenance[method] = {"derived_scores": {"path": str(path.relative_to(tmp_path)),
                              "sha256": m.base.file_sha(path), "bytes": path.stat().st_size}}
    put(out / "selection_manifest.json", {"selected_query_ids": [q["query_id"] for q in queries], "methods": provenance})
    put(out / "inputs/manifest.json", {"queries": 145})
    put(out / "protocol.json", {"methods": methods}); put(out / "results.json", {"complete": True})
    monkeypatch.setattr(m, "qualification_materials", lambda: (queries, candidates, packets, protocol))
    return queries, candidates, packets, protocol


def test_prepare_freezes_only_145_plan_without_premature_admission(cohort):
    result = m.prepare_plan()
    assert len(result[0]) == 145 and len(result[1]) == 280
    plan = m.base.read(m.ASSESS / "core_plan.json")
    assert plan["expected_primary_pairs"] == 40600 and plan["unique_nonempty_source_requests"] == 40600
    assert not (m.ASSESS / "judge_protocol.json").exists()
    assert not (m.ASSESS / "qualification_acceptance.json").exists()
    assert m.prepare_plan() == result
    assert (m.ASSESS / "sources.jsonl").resolve() == (m.QUAL_ASSESS / "sources.jsonl").resolve()


@pytest.mark.parametrize("change", ["extra", "text", "candidate_order", "stratum"])
def test_changed_cohort_never_silently_reverts_to_old_772(cohort, change):
    queries, candidates, _, _ = cohort
    changed = json.loads(json.dumps(queries))
    if change == "extra":
        changed.append({**changed[0], "query_id": "extra"})
    elif change == "text":
        changed[0]["text"] += " changed"
    elif change == "candidate_order":
        rows(m.OUT / "inputs/candidates.jsonl", candidates[::-1])
    elif change == "stratum":
        changed[1]["task_type"] = changed[0]["task_type"]
    rows(m.OUT / "inputs/queries.jsonl", changed)
    with pytest.raises(ValueError):
        m.prepare_plan()


@pytest.mark.parametrize("filename", ["scores.npy", "index.json", "status.json", "query_seconds.npy"])
def test_real_score_and_metadata_changes_are_detected_after_freeze(cohort, filename):
    m.prepare_plan()
    put(m.OUT / "results/same-ranking-0" / filename, {"finite_but_changed": 1.25})
    with pytest.raises(ValueError):
        m.prepare_plan()


def accepted(cohort, monkeypatch):
    m.prepare_plan()
    review = m.QUAL_ASSESS / "source_review.json"
    put(review, {"accepted": True, "reviewer": "Codex evidence audit; not expert gold"})
    value = {"accepted": True, "cases_per_model": 765, "scientific_gold_accuracy": None,
             "source_review_sha256": m.base.file_sha(review),
             "files": {str(review.relative_to(m.ROOT)): m.base.file_sha(review)}}
    put(m.ASSESS / "qualification_acceptance.json", value)
    def configure():
        m.base.ASSESS, m.base.RUN = m.QUAL_ASSESS, m.QUAL_RUN
        m.base.validate_explicit_scope = "frozen-v8-guard"
    monkeypatch.setattr(m.r, "configure", configure)
    return value


def test_final_protocol_binds_actual_qualification_and_retains_core_after_nested_prepare(cohort, monkeypatch):
    value = accepted(cohort, monkeypatch)
    result = m.prepare_core()
    assert m.base.OUT == m.OUT and m.base.ASSESS == m.ASSESS and m.base.RUN == m.RUN
    assert m.base.prepare is m.prepare_core and m.base.validate_explicit_scope == "frozen-v8-guard"
    assert len(m.base.prepare()[0]) == 145
    protocol = m.base.read(m.ASSESS / "judge_protocol.json")
    assert protocol == result[3] and protocol["primary_repo"] == "Qwen/Qwen3.6-27B"
    assert protocol["generation"] == cohort[3]["generation"]
    assert protocol["qualification_source_review_sha256"] == value["source_review_sha256"]
    assert protocol["input_sha256"]["inputs/queries.jsonl"] == m.base.file_sha(m.OUT / "inputs/queries.jsonl")
    assert m.base.read(m.ASSESS / "assessment_manifest.json")["expected_pairs"] == 40600


def test_missing_or_changed_review_blocks_production(cohort, monkeypatch):
    with pytest.raises(RuntimeError, match="not been completed"):
        m.prepare_core()
    accepted(cohort, monkeypatch)
    put(m.QUAL_ASSESS / "source_review.json", {"accepted": False})
    with pytest.raises(ValueError, match="qualification artifact changed"):
        m.prepare_core()


@pytest.mark.parametrize("judge,model", [("primary", "qwen"), ("secondary", "phi")])
def test_real_full_handoff_preserves_subset_model_guard_and_separate_paths(cohort, monkeypatch, judge, model):
    accepted(cohort, monkeypatch)
    python = m.FAST if judge == "primary" else m.APP
    monkeypatch.setattr(sys, "prefix", str(python.parent.parent))
    seen = {}
    class Judge:
        pass
    def backend(selected):
        seen["actual_model"] = selected
        return Judge
    monkeypatch.setattr(m.r, "backend", backend)
    def full(args):
        assert m.base.GPUJudge is Judge
        before = (m.base.OUT, m.base.ASSESS, m.base.RUN)
        assert len(m.base.prepare()[0]) == 145
        assert before == (m.base.OUT, m.base.ASSESS, m.base.RUN) == (m.OUT, m.ASSESS, m.RUN)
        assert m.base.GPUJudge is Judge and m.base.validate_explicit_scope == "frozen-v8-guard"
        assert args.batch_size == 16 and args.secondary_weights == m.r.PHI_WEIGHTS
        seen["judge"] = args.judge
    monkeypatch.setattr(m.base, "run_assessment", full)
    m.run_full(judge)
    assert seen == {"actual_model": model, "judge": judge}


def test_commands_never_dispatch_old_pipeline_or_full(cohort):
    stages = m.stages()
    assert [s[1] for s in stages] == [m.FAST, m.APP, m.FAST, m.APP]
    assert [s[2][0] for s in stages] == ["pilot", "pilot", "run", "run"]
    assert all(m.command(p, *args)[1] == str(m.SCRIPT) for _, p, args, _ in stages)
    assert all(s[3].parent == m.QUAL_RUN for s in stages[:2])
    assert all(s[3].parent == m.RUN for s in stages[2:])


@pytest.mark.parametrize("change", [None, "generic_timeout", "wrong_protocol", "quality_guard", "wrong_state"])
def test_only_exact_seven_day_same_protocol_checkpoint_is_retryable(cohort, change):
    put(m.ASSESS / "judge_protocol.json", {"frozen": True})
    status = {"state": "checkpoint_time_limit", "error_type": "TimeoutError",
              "error_message": m.SEVEN_DAY_MESSAGE, "protocol_sha256": m.base.file_sha(m.ASSESS / "judge_protocol.json")}
    if change == "generic_timeout": status["error_message"] = "GPU memory wait exceeded 50 hours"
    if change == "wrong_protocol": status["protocol_sha256"] = "changed"
    if change == "quality_guard": status["guard_reason"] = "PrimaryInvalidFractionAbove10Percent"
    if change == "wrong_state": status["state"] = "failed"
    assert m.retryable_timeout(status) is (change is None)


def test_review_wait_honors_core_stop_without_acceptance(cohort, monkeypatch):
    monkeypatch.setattr(m.r, "check_gate", lambda require_review: {"numeric_accepted": True})
    (m.RUN / "stop.requested").touch()
    updates = []
    assert m.wait_for_review(lambda **values: updates.append(values)) is False
    assert updates[-1]["state"] == "stopped"
    assert not (m.ASSESS / "qualification_acceptance.json").exists()


@pytest.mark.parametrize("relationship,expected", [("parent", True), ("self", True), ("unrelated", False), ("invalid", True)])
def test_only_verified_own_coordinator_is_exempt_from_project_mutex(cohort, monkeypatch, relationship, expected):
    pid = os.getppid() if relationship == "parent" else os.getpid() if relationship == "self" else 999991
    saved = {"pid": pid, "start_ticks": "verified"}
    put(m.RUN / "pipeline_pid.json", saved)
    monkeypatch.setattr(m.base, "alive", lambda value: value == saved and relationship != "invalid")
    if expected:
        m.assert_project_idle()
    else:
        with pytest.raises(RuntimeError, match="Another project"):
            m.assert_project_idle()


def test_old_qualification_coordinator_is_never_exempt(cohort, monkeypatch):
    saved = {"pid": 999992, "start_ticks": "verified"}
    put(m.QUAL_RUN / "pipeline_pid.json", saved)
    monkeypatch.setattr(m.base, "alive", lambda value: value == saved)
    with pytest.raises(RuntimeError, match="independent qualification"):
        m.assert_project_idle()


@pytest.mark.parametrize("initial_state,restarts", [("checkpoint_time_limit", 1), ("failed", 0)])
def test_scheduler_restarts_checkpoint_but_never_other_failure(cohort, monkeypatch, initial_state, restarts):
    put(m.ASSESS / "judge_protocol.json", {"frozen": True})
    monkeypatch.setattr(m, "assert_project_idle", lambda: None)
    monkeypatch.setattr(m, "prepare_plan", lambda: None)
    monkeypatch.setattr(m, "wait_for_review", lambda update: True)
    monkeypatch.setattr(m, "configure_core", lambda: None)
    monkeypatch.setattr(m, "verify_acceptance", lambda: {"source_review_sha256": "actual-reviewed-source"})
    monkeypatch.setattr(m, "stages", lambda: [("primary", m.FAST, ["run"], m.RUN / "status.json")])
    monkeypatch.setattr(m.base, "judge_agreement", lambda: {})
    evaluator = importlib.import_module("evaluate_frontier_automatic")
    monkeypatch.setattr(evaluator, "evaluate_run", lambda *args, **kwargs: {"state": "complete"})
    launched = []
    class Child:
        def __init__(self, *args, **kwargs):
            launched.append(self)
            self.pid = 1000 + len(launched)
            self.returncode = 1 if len(launched) == 1 else 0
            state = initial_state if len(launched) == 1 else "complete"
            put(m.RUN / "status.json", {"pid": self.pid, "state": state, "error_type": "TimeoutError",
                "error_message": m.SEVEN_DAY_MESSAGE,
                "protocol_sha256": m.base.file_sha(m.ASSESS / "judge_protocol.json")})
        def poll(self): return self.returncode
    monkeypatch.setattr(m.subprocess, "Popen", Child)
    m.pipeline()
    assert len(launched) == 1 + restarts
    state = m.base.read(m.RUN / "pipeline_status.json")
    assert state["state"] == ("complete" if restarts else "failed")
