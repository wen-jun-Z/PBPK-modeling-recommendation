"""CPU integration of development disclosure and the active request guard."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
v = importlib.import_module("run_frontier_scoring_citation_complete")
r, fast = v.r, v.fast


@pytest.fixture(autouse=True)
def restore_modules():
    snapshots = [(m, dict(vars(m))) for m in (r, fast, r.shared, r.engine, r.base, r.utilities)]
    environment = dict(os.environ)
    yield
    for module, original in snapshots:
        for name, value in original.items(): setattr(module, name, value)
    os.environ.clear(); os.environ.update(environment)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    out = tmp_path / "experiments"
    assess, run = out / "assessment_citation_complete_v8", tmp_path / "runtime/frontier_scoring_reviewed_v8"
    assess.mkdir(parents=True); run.mkdir(parents=True)
    for module in (v, fast, r): monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(fast, "FAST", tmp_path / "runtime/judge-fast-env/bin/python")
    for name, value in (("OUT", out), ("ASSESS", assess), ("RUN", run), ("PROFILE", "v8"),
            ("CONFIG_SHA", "v8-config"), ("SYSTEM", "v8-complete-citation-chain")):
        monkeypatch.setattr(r, name, value)
    old_record = tmp_path / "runtime/v7_record.json"; put(old_record, {"unchanged": True})
    snapshot = tmp_path / "runtime/v7_snapshot.json"
    put(snapshot, {"files": {"runtime/v7_record.json": r.base.file_sha(old_record)}})
    config = {"citation_runner_file": "scripts/run_frontier_scoring_citation_complete.py",
        "citation_runner_sha256": r.base.file_sha(Path(v.__file__)), "regression_disclosure": v.DISCLOSURE,
        "request_scope_guard_file": "scripts/frontier_request_scope_guard.py", "request_scope_guard_sha256": "guard-sha",
        "request_scope_guard_protocol": {"version": "actual-request-guard", "labels_changed": False},
        "v7_failure_snapshot_file": "runtime/v7_snapshot.json", "v7_failure_snapshot_file_sha256": r.base.file_sha(snapshot),
        "fresh_origin_assessment_directory": "assessment_precedence_v5",
        "reasoned_backend_file": "scripts/frontier_reasoned_fast_judge.py", "reasoned_backend_sha256": "backend-sha",
        "reasoning_generation_protocol": {"thinking_budget": 1024}, "legacy_manifest_sha256": "legacy", "fresh_manifest_sha256": "fresh",
        "fast_environment_manifest_sha256": "env-sha", "v6_control_comparison_file_sha256": "v6-sha"}
    monkeypatch.setattr(r, "CONFIG", config)
    calls = []
    def guard(answer, query, refs, model_compounds=None):
        calls.append((answer, query, refs, model_compounds))
        return {"applied": True, "guard_version": "actual-request-guard"}
    module = SimpleNamespace(guard_protocol=lambda: {"version": "actual-request-guard", "labels_changed": False},
        guard_factory=lambda original: guard)
    def imported(file, *args):
        if file == config["request_scope_guard_file"]: return module
        return SimpleNamespace(generation_protocol=lambda: {"thinking_budget": 1024}, ReasonedGPUJudge="real-qwen-fast")
    monkeypatch.setattr(r.shared, "import_frozen", imported)
    monkeypatch.setattr(r.shared, "assert_prior_runs_idle", lambda: None)
    monkeypatch.setattr(fast, "verify_bindings", lambda: ({"files": {"kernel": "sha"}}, {"v6_control_count": 64}))
    v.install(); v.verify_bindings()
    return assess, run, calls


def test_commands_keep_actual_model_interpreters_and_new_wrapper(setup):
    v.configure()
    assert [s[1] for s in r.stages()] == [fast.FAST, r.APP, fast.FAST, r.APP]
    for _, python, args, _ in r.stages():
        assert r.command(python, *args) == [str(python), str(Path(v.__file__).resolve()), "--profile", "v8", *args]
    assert r.engine.command(r.APP, "pipeline")[1] == str(Path(v.__file__).resolve())


def test_nested_configure_retains_guard_backend_and_original_gate(setup):
    gate = r.check_gate
    phi = r.PLAIN_BACKEND
    v.configure()
    for _ in range(3):
        r.shared.configure()
        assert r.base.validate_explicit_scope is v.GUARD
        assert r.engine.prepare is v.prepare and r.engine.command is v.command
        assert r.engine.check_gate is gate
        assert r.engine.backend("qwen") == "real-qwen-fast" and r.PLAIN_BACKEND is phi
        assert os.environ["USE_HUB_KERNELS"] == "0"


def test_actual_pilot_and_replay_paths_use_injected_guard(setup):
    _, _, calls = setup
    v.configure()
    case = {"case_id": "one", "query": {"text": "Predict AUC", "task_type": "pk_prediction"},
        "packet": {"refs": [{"quote": "Predicts AUC"}], "body": {"model_compounds": []}}}
    output = '[{"i":1,"g":3,"a":"Y","e":[1]}]'
    assert r.engine.validation_error(output, case) is None
    replay = importlib.import_module("frontier_judge_validation_v3")._replay([{"case_id": "one", "output": output}], [case])
    assert replay["final_valid_count"] == 1 and len(calls) == 2
    assert all(call[0]["grade"] == 3 and call[0]["evidence_numbers"] == [1] for call in calls)


def test_real_request_guard_fixes_endpoint_only_case_and_keeps_curve_requirement(setup, monkeypatch):
    module = importlib.import_module("frontier_request_scope_guard")
    actual = module.guard_factory(v.ORIGINAL_SCOPE_GUARD)
    monkeypatch.setattr(v, "GUARD", actual)
    v.configure()
    case = {"case_id": "endpoint", "query": {"text": "Predict AUC and Cmax in adults.",
        "task_type": "plasma_concentration_time_prediction", "features": {"endpoints": ["auc", "cmax"]}},
        "packet": {"refs": [{"quote": "This PBPK model predicts AUC and Cmax in healthy adults."}],
            "body": {"model_compounds": ["DrugX"]}}}
    before = copy.deepcopy(case)
    output = '[{"i":1,"g":3,"a":"Y","e":[1]}]'
    parsed = r.base.parse_answers(output, [1], case["packet"]["refs"])[1]
    with pytest.raises(ValueError, match="main_task"):
        v.ORIGINAL_SCOPE_GUARD(parsed, case["query"], case["packet"]["refs"], ["DrugX"])
    assert r.engine.validation_error(output, case) is None
    replay = importlib.import_module("frontier_judge_validation_v3")._replay([{"case_id": "endpoint", "output": output}], [case])
    assert replay["final_valid_count"] == 1 and case == before
    case["query"]["text"] += " Also predict the concentration-time curve."
    assert "main_task" in r.engine.validation_error(output, case)
    case = copy.deepcopy(before)
    case["packet"]["refs"][0]["quote"] = "This PBPK model predicts AUC in healthy adults."
    assert r.engine.validation_error(output, case) is not None


def test_actual_prepare_chain_freezes_truthful_post_development_protocol(setup, monkeypatch):
    assess, _, _ = setup
    origin = r.OUT / "assessment_precedence_v5/validation/fresh"
    put(origin / "cases.jsonl", {}); put(origin / "manifest.json", {})
    def generic_prepare():
        r.shared.configure()
        assert r.base.validate_explicit_scope is v.GUARD
        p = {"primary_repo": "phi", "secondary_repo": "qwen", "primary_revision": "p", "secondary_revision": "q",
            "primary_weights_manifest_sha256": "pw", "secondary_weights_manifest_sha256": "qw", "implementation_sha256": {},
            "generation": {"max_new_tokens": 96}, "execution_environments": {"phi": {"executable": str(r.APP)}, "qwen": {"executable": str(r.utilities.MODERN)}},
            "scope_guard": {"version": "old"}, "validation_policy": {"interpretation": "new controls"},
            "judge_development": {"v3_validation": "separate fresh controls", "reviewed_profile": {"new_control_count": 32,
                "fresh_definition": "Frozen after prompt and before this profile's first generation"}}}
        r.utilities.freeze(assess / "judge_protocol.json", p)
        stored = r.base.read(assess / "judge_protocol.json")
        assert stored["validation_policy"]["independent_control_count"] == 0
        r.utilities.freeze(assess / "assessment_manifest.json", {"protocol_sha256": r.base.file_sha(assess / "judge_protocol.json")})
        return [], [], {}, p
    monkeypatch.setattr(r, "SHARED_PREPARE", generic_prepare)
    before = fast.FREEZE
    result = r.prepare()[3]
    assert result == r.base.read(assess / "judge_protocol.json")
    assert result["primary_repo"] == "qwen" and result["secondary_repo"] == "phi"
    assert result["scope_guard"] == v.GUARD_PROTOCOL and result["validation_policy"]["interpretation"] == v.DISCLOSURE
    assert result["judge_development"]["reviewed_profile"]["new_control_count"] == 0
    assert "previously unexposed" not in json.dumps(result) and "Frozen after prompt" not in json.dumps(result)
    assert result["generation"]["primary_reasoning"] == {"thinking_budget": 1024}
    assert r.prepare()[3] == result and fast.FREEZE is before


@pytest.mark.parametrize("judge,model", [("primary", "qwen"), ("secondary", "phi")])
def test_production_handoff_preserves_scope_guard_and_real_model(setup, monkeypatch, judge, model):
    seen = {}
    def prepare():
        r.shared.configure()
        assert r.base.validate_explicit_scope is v.GUARD
    monkeypatch.setattr(r, "prepare", prepare)
    monkeypatch.setattr(r, "check_gate", lambda: {"accepted": True})
    def backend(actual):
        class Judge:
            def __init__(self, *args): seen["model"] = actual
        return Judge
    monkeypatch.setattr(r, "backend", backend)
    def full(args):
        before = r.base.GPUJudge
        r.base.prepare()
        assert r.base.GPUJudge is before
        r.base.GPUJudge({})
        seen["guard"] = r.base.validate_explicit_scope
    monkeypatch.setattr(r.base, "run_assessment", full)
    r.run_full(judge)
    assert seen == {"model": model, "guard": v.GUARD}


def test_failed_snapshot_cannot_change_silently(setup):
    put(v.ROOT / "runtime/v7_record.json", {"changed": True})
    with pytest.raises(ValueError, match="failed run artifact changed"): v.verify_bindings()


def test_frozen_guard_protocol_and_development_disclosure_required(setup):
    r.CONFIG["regression_disclosure"] = "Independent blind controls"
    with pytest.raises(ValueError, match="disclosure"): v.verify_bindings()
    r.CONFIG["regression_disclosure"] = v.DISCLOSURE
    r.CONFIG["request_scope_guard_protocol"] = {"version": "other"}
    with pytest.raises(ValueError, match="guard protocol changed"): v.verify_bindings()


def test_summary_replaces_inherited_fresh_and_prompt_claims(setup):
    folder = v.ROOT / "experiments/model_recommendation/results_summary"
    folder.mkdir(parents=True); (folder / "README.md").write_text("Existing results\n")
    v.configure(); v.publish_summary()
    text = (folder / "README.md").read_text()
    assert "All 192 synthetic controls are development regression" in text and "573 source cases" in text
    assert "not previously submitted to the model" not in text and "Prompts remain v5" not in text
    assert "scripts/run_frontier_scoring_citation_complete.py --profile v8" in text
