"""CPU checks of real role names, recursive injection and isolated reasoning."""
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from frozen_frontier_sources import historical_sha

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
v = importlib.import_module("run_frontier_scoring_reasoned")
REAL_ROOT = Path(v.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_modules():
    engine = {k: getattr(v.engine, k) for k in v.shared.INJECTED}
    shared = {k: getattr(v.shared, k) for k in ("PROFILE", "CONFIG", "CONFIG_SHA", "SYSTEM", "ASSESS", "RUN", "configure")}
    base = {k: getattr(v.base, k) for k in ("SYSTEM", "ASSESS", "RUN", "prepare", "GPUJudge", "download_primary")}
    own = {k: getattr(v, k) for k in ("PROFILE", "CONFIG", "CONFIG_SHA", "SYSTEM", "ASSESS", "RUN")}
    old_freeze, old_assess = v.utilities.freeze, v.utilities.ASSESS
    yield
    for key, value in engine.items(): setattr(v.engine, key, value)
    for key, value in shared.items(): setattr(v.shared, key, value)
    for key, value in base.items(): setattr(v.base, key, value)
    for key, value in own.items(): setattr(v, key, value)
    v.utilities.freeze, v.utilities.ASSESS = old_freeze, old_assess


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


@pytest.fixture
def paths(tmp_path, monkeypatch):
    out = tmp_path / "experiments"
    assess, run = out / "assessment_reasoned_v6", tmp_path / "runtime/deployment/frontier_scoring_reviewed_v6"
    assess.mkdir(parents=True); run.mkdir(parents=True)
    for key, value in (("ROOT", tmp_path), ("OUT", out), ("ASSESS", assess), ("RUN", run),
                       ("PROFILE", "v6"), ("CONFIG_SHA", "config-sha"), ("SYSTEM", "unchanged v5")):
        monkeypatch.setattr(v, key, value)
    monkeypatch.setattr(v, "CONFIG", {"fresh_origin_assessment_directory": "assessment_precedence_v5",
        "reasoned_backend_file": "scripts/frontier_reasoned_judge.py", "reasoned_backend_sha256": "backend-sha",
        "reasoning_generation_protocol": {"thinking_budget": 1024}, "legacy_manifest_sha256": "legacy-sha", "fresh_manifest_sha256": "fresh-sha"})
    monkeypatch.setattr(v.shared, "assert_prior_runs_idle", lambda: None)
    return assess, run


def test_recursive_generic_configure_always_reinjects_real_reasoned_roles(paths):
    v.configure()
    for _ in range(3):
        v.shared.configure()
        assert v.engine.prepare is v.prepare and v.engine.backend is v.backend
        assert v.engine.check_gate is v.check_gate and v.engine.command is v.command
        assert v.engine.verify_source_review is v.verify_source_review
        assert v.base.ASSESS == v.ASSESS and v.base.SYSTEM == v.SYSTEM


def test_protocol_transformed_before_freeze_and_return_matches_disk_after_recursive_prepare(paths, monkeypatch):
    assess, run = paths
    origin = v.OUT / "assessment_precedence_v5/validation/fresh"
    put(origin / "cases.jsonl", {}); put(origin / "manifest.json", {})
    monkeypatch.setattr(v.shared, "import_frozen", lambda *args: SimpleNamespace(generation_protocol=lambda: {"thinking_budget": 1024}))
    old = {"primary_repo": "phi", "secondary_repo": "qwen", "primary_revision": "phi-r", "secondary_revision": "qwen-r",
        "primary_weights_manifest_sha256": "phi-w", "secondary_weights_manifest_sha256": "qwen-w",
        "generation": {"max_new_tokens": 96}, "implementation_sha256": {"run_frontier_scoring_reviewed.py": "old-generic"}}
    def recursive_prepare():
        v.shared.configure()
        assert v.engine.backend is v.backend and v.engine.check_gate is v.check_gate
        v.utilities.freeze(assess / "judge_protocol.json", old)
        assert v.base.read(assess / "judge_protocol.json")["primary_repo"] == "qwen"
        v.utilities.freeze(assess / "assessment_manifest.json", {"protocol_sha256": v.base.file_sha(assess / "judge_protocol.json")})
        return [1], [2], {3: 4}, old
    monkeypatch.setattr(v, "SHARED_PREPARE", recursive_prepare)
    result = v.prepare()
    assert result[3] == v.base.read(assess / "judge_protocol.json")
    assert result[3]["primary_repo"] == "qwen" and result[3]["secondary_repo"] == "phi"
    assert result[3]["primary_weights_manifest_sha256"] == "qwen-w"
    assert result[3]["generation"]["primary_reasoning"] == {"thinking_budget": 1024}
    assert result[3]["implementation_sha256"][Path(v.__file__).name] == v.base.file_sha(Path(v.__file__))
    assert v.engine.prepare is v.prepare and v.engine.backend is v.backend
    assert v.utilities.freeze is v.ORIGINAL_FREEZE
    assert v.prepare()[3] == result[3]


def test_freeze_hook_is_restored_even_when_shared_prepare_fails(paths, monkeypatch):
    origin = v.OUT / "assessment_precedence_v5/validation/fresh"
    put(origin / "cases.jsonl", {}); put(origin / "manifest.json", {})
    monkeypatch.setattr(v.shared, "import_frozen", lambda *args: SimpleNamespace(generation_protocol=lambda: {"thinking_budget": 1024}))
    def failed(): raise ValueError("fixture failure")
    monkeypatch.setattr(v, "SHARED_PREPARE", failed)
    with pytest.raises(ValueError, match="fixture failure"): v.prepare()
    assert v.utilities.freeze is v.ORIGINAL_FREEZE


@pytest.fixture
def pilot(paths, monkeypatch):
    assess, run = paths
    all_cases = v.base.read_rows(REAL_ROOT / "experiments/model_recommendation/frontier_comparison_20260914/assessment_evidence_first_v3/validation/cases.jsonl")
    legacy = [copy.deepcopy(next(c for c in all_cases if c["strata"]["validation_role"] == r and c["expected"]["grades"] == [3])) for r in v.CONTROL_ROLES]
    legacy.append(copy.deepcopy(next(c for c in all_cases if c["kind"] == "real_source_pair")))
    fresh = [copy.deepcopy(legacy[0])]; fresh[0]["case_id"] = "new-fresh"; fresh[0]["query"]["query_id"] = "new-fresh-query"
    put(assess / "validation/manifest.json", {"cohort": "legacy"}); put(assess / "validation/fresh/manifest.json", {"cohort": "fresh"})
    v.base.append_rows(assess / "validation/cases.jsonl", legacy); v.base.append_rows(assess / "validation/fresh/cases.jsonl", fresh)
    put(assess / "judge_protocol.json", {"primary_repo": "qwen", "secondary_repo": "phi"})
    for model in ("qwen", "phi"):
        path = assess / f"{model}_weights.json"; put(path, {"repo": model, "revision": "pinned"})
        monkeypatch.setattr(v, "QWEN_WEIGHTS" if model == "qwen" else "PHI_WEIGHTS", path)
        monkeypatch.setattr(v.shared, "QWEN_WEIGHTS" if model == "qwen" else "PHI_WEIGHTS", path)
    monkeypatch.setattr(v, "prepare", lambda: None)
    monkeypatch.setattr(v.utilities, "verify_weights", lambda path: v.base.read(path))
    validator = importlib.import_module("frontier_judge_validation_v3")
    monkeypatch.setattr(validator, "evaluate_validation_v3", lambda rows, cases: {"complete": len(rows) == len(cases)})
    return legacy, fresh


def make_backend(monkeypatch, legacy, fresh, failing=(), source_repair=False):
    mapping = {c["query"]["query_id"]: c for c in [*legacy, *fresh]}
    calls, loads = [], []
    class Backend:
        def __init__(self, weights, *args): loads.append(weights["repo"])
        def select_batch_size(self, tasks, maximum): return len(tasks)
        def generate(self, tasks, max_new_tokens=96, repairs=None):
            outputs = []
            for packet, queries in tasks:
                assert "expected" not in queries[0]
                case = mapping[queries[0]["query_id"]]; calls.append((case["case_id"], max_new_tokens, bool(repairs)))
                if case["case_id"] in failing or case["expected"] is None:
                    output = '[{"i":1,"g":null,"a":"U","e":[]}]'
                else:
                    e = case["expected"]
                    output = json.dumps([{"i": 1, "g": e["grades"][0], "a": {True: "Y", False: "N", None: "U"}[e["documented_applicability"]],
                        "e": sorted({x[0] for x in e["required_support_any_of"]})}])
                if source_repair and case["expected"] is None and not repairs: output = "["
                outputs.append(output)
            return outputs, {"wall_seconds": 1., "reasoning_artifact": {"path": "scoped/reference.jsonl", "offset": 0, "length": 40,
                "sha256": "artifact-hash", "task_count": len(tasks)}, "thinking_token_counts": [13]*len(tasks)}
    monkeypatch.setattr(v, "backend", lambda model: Backend)
    return calls, loads


def test_qwen_is_actual_early_stopping_primary_not_phi(pilot, monkeypatch):
    legacy, fresh = pilot
    calls, loads = make_backend(monkeypatch, legacy, fresh, failing={legacy[0]["case_id"]})
    v.pilot("qwen")
    assert loads == ["qwen"] and [r[0] for r in calls] == [legacy[0]["case_id"]]
    state = v.base.read(v.RUN / "pilot_qwen_status.json")
    assert state["primary"] and state["state"] == "stopped_on_control_gate"
    assert not (v.ASSESS / "validation/phi_records.jsonl").exists()
    with pytest.raises(RuntimeError, match="identical restart"): v.execute("start")


def test_phi_reference_failure_is_preserved_without_stopping_other_groups(pilot, monkeypatch):
    legacy, fresh = pilot
    calls, loads = make_backend(monkeypatch, legacy, fresh, failing={legacy[0]["case_id"]})
    v.pilot("phi")
    assert loads == ["phi"] and len(calls) == len(legacy)+len(fresh)
    assert v.base.read(v.RUN / "pilot_phi_status.json")["state"] == "complete"
    gate = v.base.read(v.ASSESS / "validation" / f"phi_{v.CONTROL_ROLES[0]}_gate.json")
    assert gate["absolute_failed_checks"] and not gate["primary"]


def test_reasoning_artifact_references_survive_raw_final_repair_and_independent_cohorts(pilot, monkeypatch):
    legacy, fresh = pilot
    calls, loads = make_backend(monkeypatch, legacy, fresh, source_repair=True)
    v.pilot("qwen")
    assert [x[0] for x in calls[:4]] == [c["case_id"] for c in legacy[:3]]+[fresh[0]["case_id"]]
    rows, raw = v.shared.read_pilot("qwen", "legacy", legacy)
    assert rows[legacy[0]["case_id"]]["model_repo"] == "qwen"
    assert rows[legacy[0]["case_id"]]["metrics"]["reasoning_artifact"]["sha256"] == "artifact-hash"
    assert len(rows[legacy[-1]["case_id"]]["attempts"]) == 2
    assert all(a["batch_metrics"]["reasoning_artifact"] for a in rows[legacy[-1]["case_id"]]["attempts"])
    assert len(v.base.load_committed(v.ASSESS / "validation/fresh/qwen_records.jsonl")) == len(fresh)
    monkeypatch.setattr(v, "backend", lambda model: pytest.fail("Completed reasoning must not regenerate on resume"))
    v.pilot("qwen")


def test_qwen_fresh_failure_stops_before_source_tasks(pilot, monkeypatch):
    legacy, fresh = pilot
    calls, loads = make_backend(monkeypatch, legacy, fresh, failing={fresh[0]["case_id"]})
    v.pilot("qwen")
    assert len(calls) == 4 and v.base.read(v.RUN / "pilot_qwen_status.json")["failed_control_group"] == "fresh32"


def test_true_stage_order_python_and_progress_paths(paths):
    stages = v.stages()
    assert [x[0] for x in stages] == ["pilot_qwen", "pilot_phi", "primary", "secondary"]
    assert [x[1] for x in stages] == [v.MODERN, v.APP, v.MODERN, v.APP]
    assert stages[0][3].name == "pilot_qwen_status.json"
    assert v.command(v.MODERN, "pilot", "--model", "qwen")[2:] == ["--profile", "v6", "pilot", "--model", "qwen"]


@pytest.mark.parametrize("judge,model", [("primary", "qwen"), ("secondary", "phi")])
def test_full_handoff_preserves_roles_across_recursive_configuration(paths, monkeypatch, judge, model):
    seen = {}
    def gate():
        v.shared.configure()
        assert v.engine.backend is v.backend
        return {"accepted": True}
    monkeypatch.setattr(v, "check_gate", gate)
    def backend(name):
        seen["model"] = name
        class Actual:
            def __init__(self, *args): seen["constructed"] = name
        return Actual
    monkeypatch.setattr(v, "backend", backend)
    def nested_prepare():
        v.shared.configure()
        assert v.engine.backend is v.backend and v.engine.check_gate is v.check_gate
        return [], [], {}, {}
    monkeypatch.setattr(v, "prepare", nested_prepare)
    monkeypatch.setattr(v.utilities, "verify_weights", lambda path: seen.update(primary_manifest=path))
    def run(args):
        seen["args"] = args
        before = v.base.GPUJudge
        v.base.prepare()
        assert v.base.GPUJudge is before
        v.base.GPUJudge({}, 36)
        v.base.download_primary()
    monkeypatch.setattr(v.base, "run_assessment", run)
    v.run_full(judge)
    assert seen["model"] == seen["constructed"] == model and seen["primary_manifest"] == v.QWEN_WEIGHTS
    assert seen["args"].secondary_weights == v.PHI_WEIGHTS
    assert v.base.prepare is v.prepare


def test_reasoned_backend_only_for_actual_qwen_name(paths, monkeypatch):
    marker = object()
    monkeypatch.setattr(v.shared, "import_frozen", lambda *args: SimpleNamespace(ReasonedGPUJudge=marker))
    monkeypatch.setattr(v, "PLAIN_BACKEND", lambda model: "plain-"+model)
    assert v.backend("qwen") is marker and v.backend("phi") == "plain-phi"


@pytest.fixture
def scheduler(paths, monkeypatch):
    monkeypatch.setattr(v, "prepare", lambda: None)
    monkeypatch.setattr(v, "publish_summary", lambda: None)
    monkeypatch.setattr(v.signal, "signal", lambda *args: None)
    monkeypatch.setattr(v.base, "stop_child", lambda child: None)
    monkeypatch.setattr(v, "check_gate", lambda require_review=True: {"numeric_accepted": True, "accepted": True})
    monkeypatch.setattr(v.engine, "wait_for_source_review", lambda update: True)
    monkeypatch.setattr(v, "verify_source_review", lambda: (True, "source-review-sha"))
    put(v.ASSESS / "judge_protocol.json", {})
    return paths


def test_pipeline_failed_qwen_pilot_does_not_start_phi_or_full(scheduler, monkeypatch):
    calls = []
    class Child:
        pid, returncode = 200, 0
        def __init__(self, command, **kwargs):
            calls.append(command)
            put(v.RUN / "pilot_qwen_status.json", {"pid": self.pid, "state": "stopped_on_control_gate", "completed": 64,
                "guard_reason": "AbsoluteControlAdmissionFailed:legacy:first"})
        def poll(self): return self.returncode
    monkeypatch.setattr(v.subprocess, "Popen", Child)
    v.pipeline()
    assert len(calls) == 1 and calls[0][-1] == "qwen"
    status = v.base.read(v.RUN / "pipeline_status.json")
    assert status["state"] == "failed" and status["failed_stage"] == "pilot_qwen" and status["completed"] == 64


def test_pipeline_complete_uses_actual_model_identities_and_strict_evaluator(scheduler, monkeypatch):
    calls, evaluations = [], []
    class Child:
        returncode = 0
        def __init__(self, command, **kwargs):
            calls.append(command); self.pid = 200+len(calls)
            name = command[-1]
            path = v.RUN / (f"pilot_{name}_status.json" if name in {"phi", "qwen"} else "status.json")
            put(path, {"pid": self.pid, "state": "complete", "completed": 765,
                "committed_pairs": 216160 if name == "primary" else 10771})
        def poll(self): return self.returncode
    monkeypatch.setattr(v.subprocess, "Popen", Child)
    monkeypatch.setattr(v.base, "judge_agreement", lambda: {"paired": 10771})
    evaluator = importlib.import_module("evaluate_frontier_automatic")
    monkeypatch.setattr(evaluator, "evaluate_run", lambda *args, **kw: evaluations.append(kw) or {"state": "complete"})
    v.pipeline()
    assert [c[-1] for c in calls] == ["qwen", "phi", "primary", "secondary"]
    assert [c[0] for c in calls] == [str(v.MODERN), str(v.APP), str(v.MODERN), str(v.APP)]
    assert evaluations == [{"require_complete": True}, {"require_complete": True}]
    result = v.base.read(v.ASSESS / "judge_agreement.json")
    assert result["primary_repo"] == "Qwen/Qwen3.6-27B" and result["secondary_repo"] == "microsoft/phi-4"
    assert v.base.read(v.RUN / "pipeline_status.json")["state"] == "complete"


def test_source_review_requires_true_selected_role_names(paths, monkeypatch):
    monkeypatch.setattr(v.shared, "verify_source_review", lambda: (True, "verified-cohorts"))
    put(v.ASSESS / "source_review.json", {"primary": "phi", "secondary": "qwen"})
    with pytest.raises(ValueError, match="actual Qwen"): v.verify_source_review()
    put(v.ASSESS / "source_review.json", {"primary": "qwen", "secondary": "phi"})
    assert v.verify_source_review() == (True, "verified-cohorts")


def test_frozen_helpers_remain_unchanged():
    assert historical_sha("run_frontier_scoring_reviewed.py") == "6d3f42c3ab7d7411e2df6880aa7a8175f4334307f3bbc3c505b08e7b52c6f57b"
    assert historical_sha("run_frontier_scoring_v3.py") == "f389c9093640325320f5ed7306fd0151f87f88506878bbb5dda1e45d909cc2f2"


@pytest.mark.parametrize("qwen_score,phi_score,accepted", [(1., .875, True), (.89, .875, False), (.95, 1., False)])
def test_all_group_gates_use_real_qwen_primary_and_phi_reference(paths, monkeypatch, qwen_score, phi_score, accepted):
    monkeypatch.setattr(v, "prepare", lambda: None)
    monkeypatch.setattr(v.shared, "cohorts", lambda: ([{"case_id": "old"}], [{"case_id": "new"}]))
    def read(model, cohort, cases):
        rows = {c["case_id"]: {"case_id": c["case_id"], "model": model} for c in cases}
        return rows, rows
    monkeypatch.setattr(v.shared, "read_pilot", read)
    def metric(model):
        return {"complete": True, "controlled_semantic_accuracy": qwen_score if model == "qwen" else phi_score,
            "controlled_citation_support_accuracy": 1., "citation_numbers_valid_fraction": 1., "raw_format_valid_fraction": 1.}
    def report(rows, cases):
        model = rows[0]["model"]
        return {"complete": True, "admission_by_role": {r: metric(model) for r in v.CONTROL_ROLES},
            "by_role": {"all": {"rows": [{"schema_valid": True, "citation_numbers_valid": True}]}},
            "final_validation_by_role": {"prospective_source_coverage": {"final_valid_fraction": 1.}},
            "coverage_sequence": {"complete": True, "any_full_window_above10percent": False}}
    validator = importlib.import_module("frontier_judge_validation_v3")
    monkeypatch.setattr(validator, "evaluate_validation_v3", report)
    monkeypatch.setattr(v.shared, "control_metric", lambda rows, cases: metric(rows[0]["model"]))
    monkeypatch.setattr(v, "verify_source_review", lambda: (True, "review"))
    put(v.ASSESS / "judge_protocol.json", {})
    result = v.check_gate()
    assert result["accepted"] is accepted
    assert result["primary"] == "Qwen/Qwen3.6-27B" and result["secondary"] == "microsoft/phi-4"
    assert result["control_groups"][v.CONTROL_ROLES[0]]["qwen"]["controlled_semantic_accuracy"] == qwen_score
    assert v.engine.prepare is v.prepare and v.engine.backend is v.backend
