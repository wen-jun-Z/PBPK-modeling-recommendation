"""CPU coverage for the isolated kernel wrapper and nested frozen handoff."""
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
v = importlib.import_module("run_frontier_scoring_fast")
r = v.reasoned
REAL_ROOT = Path(v.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_modules():
    modules = (r, r.shared, r.engine, r.base, r.utilities)
    snapshots = [(m, dict(vars(m))) for m in modules]
    yield
    for module, before in snapshots:
        for name, value in before.items(): setattr(module, name, value)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    out = tmp_path / "experiments"
    assess = out / "assessment_reasoned_fast_v7"
    run = tmp_path / "runtime/deployment/frontier_scoring_reviewed_v7"
    assess.mkdir(parents=True); run.mkdir(parents=True)
    fast = tmp_path / "runtime/judge-fast-env/bin/python"
    fast.parent.mkdir(parents=True); fast.write_text("fixture interpreter")
    artifact = tmp_path / "runtime/judge-fast-env/pyvenv.cfg"
    artifact.write_text("fixture isolated overlay")
    env = tmp_path / "runtime/fast_environment.json"
    put(env, {"interpreter": "runtime/judge-fast-env/bin/python",
        "files": {"runtime/judge-fast-env/pyvenv.cfg": r.base.file_sha(artifact)}})
    comparison = tmp_path / "runtime/v6_comparison.json"
    put(comparison, {"control_count": 64, "predictions_reused": False})
    old = tmp_path / "scripts/run_frontier_scoring_reasoned.py"
    old.parent.mkdir(); old.write_bytes((REAL_ROOT / "scripts/run_frontier_scoring_reasoned.py").read_bytes())
    monkeypatch.setattr(v, "ROOT", tmp_path); monkeypatch.setattr(v, "FAST", fast)
    for name, value in (("ROOT", tmp_path), ("OUT", out), ("ASSESS", assess), ("RUN", run),
            ("PROFILE", "v7"), ("CONFIG_SHA", "fixture-config"), ("SYSTEM", "same-v5-system")):
        monkeypatch.setattr(r, name, value)
    config = {"fast_runner_file": "scripts/run_frontier_scoring_fast.py",
        "fast_runner_sha256": r.base.file_sha(Path(v.__file__)),
        "fast_environment_manifest": "runtime/fast_environment.json",
        "fast_environment_manifest_sha256": r.base.file_sha(env),
        "v6_control_comparison_file": "runtime/v6_comparison.json",
        "v6_control_comparison_file_sha256": r.base.file_sha(comparison),
        "fresh_origin_assessment_directory": "assessment_precedence_v5",
        "reasoned_backend_file": "scripts/frontier_reasoned_fast_judge.py", "reasoned_backend_sha256": "backend-sha",
        "reasoning_generation_protocol": {"thinking_budget": 1024},
        "legacy_manifest_sha256": "legacy", "fresh_manifest_sha256": "fresh"}
    monkeypatch.setattr(r, "CONFIG", config)
    monkeypatch.setattr(r.shared, "assert_prior_runs_idle", lambda: None)
    v.install()
    return assess, run, artifact


def test_actual_stage_commands_keep_qwen_fast_and_phi_app(setup):
    v.configure()
    stages = r.stages()
    assert [x[0] for x in stages] == ["pilot_qwen", "pilot_phi", "primary", "secondary"]
    assert [x[1] for x in stages] == [v.FAST, r.APP, v.FAST, r.APP]
    for _, executable, arguments, _ in stages:
        command = r.command(executable, *arguments)
        assert command == [str(executable), str(Path(v.__file__).resolve()), "--profile", "v7", *arguments]
    assert r.engine.command(r.APP, "pipeline")[1] == str(Path(v.__file__).resolve())


def test_nested_configuration_cannot_reset_backend_or_wrapper(setup, monkeypatch):
    marker = object()
    monkeypatch.setattr(r.shared, "import_frozen", lambda *a: SimpleNamespace(ReasonedGPUJudge=marker))
    phi_backend = r.PLAIN_BACKEND
    v.configure()
    for _ in range(3):
        r.shared.configure()
        assert r.engine.prepare is v.prepare and r.engine.command is v.command
        assert r.engine.backend("qwen") is marker
        assert r.PLAIN_BACKEND is phi_backend
        assert r.engine.check_gate is r.check_gate
        assert r.MODERN == r.utilities.MODERN == v.FAST


def test_real_initialize_environment_cleanup_then_execute_restores_local_dispatch(setup, monkeypatch):
    original = dict(os.environ)
    def executed(*args):
        assert os.environ["USE_HUB_KERNELS"] == "0"
        assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"
    monkeypatch.setattr(v, "EXECUTE", executed)
    monkeypatch.setattr(r.base, "environment", lambda: {})
    try:
        v.configure()
        assert os.environ["USE_HUB_KERNELS"] == "0"
        r.engine.initialize()
        assert "USE_HUB_KERNELS" not in os.environ
        v.execute("start")
    finally:
        os.environ.clear(); os.environ.update(original)


def test_real_v6_prepare_chain_freezes_fast_protocol_before_manifest(setup, monkeypatch):
    assess, _, _ = setup
    origin = r.OUT / "assessment_precedence_v5/validation/fresh"
    put(origin / "cases.jsonl", {}); put(origin / "manifest.json", {})
    monkeypatch.setattr(r.shared, "import_frozen", lambda *a: SimpleNamespace(generation_protocol=lambda: {"thinking_budget": 1024}))
    def inner_prepare():
        r.shared.configure()
        assert r.engine.prepare is v.prepare
        protocol = {"primary_repo": "phi", "secondary_repo": "qwen", "primary_revision": "p", "secondary_revision": "q",
            "primary_weights_manifest_sha256": "pw", "secondary_weights_manifest_sha256": "qw",
            "implementation_sha256": {}, "generation": {"max_new_tokens": 96},
            "execution_environments": {"phi": {"executable": str(r.APP)}, "qwen": {"executable": str(r.utilities.MODERN)}}}
        r.utilities.freeze(assess / "judge_protocol.json", protocol)
        stored = r.base.read(assess / "judge_protocol.json")
        assert stored["primary_repo"] == "qwen" and stored["fast_environment_manifest"]
        r.utilities.freeze(assess / "assessment_manifest.json", {"protocol_sha256": r.base.file_sha(assess / "judge_protocol.json")})
        return [1], [2], {"packet": 3}, protocol
    monkeypatch.setattr(r, "SHARED_PREPARE", inner_prepare)
    old_hook = r.ORIGINAL_FREEZE
    result = r.prepare()
    assert result[3] == r.base.read(assess / "judge_protocol.json")
    assert result[3]["primary_weights_manifest_sha256"] == "qw"
    assert result[3]["generation"]["primary_reasoning"] == {"thinking_budget": 1024}
    assert result[3]["execution_environments"]["phi"]["executable"] == str(r.APP)
    assert result[3]["implementation_sha256"][Path(v.__file__).name] == r.base.file_sha(Path(v.__file__))
    assert r.ORIGINAL_FREEZE is old_hook
    assert r.prepare()[3] == result[3]


def test_environment_and_disclosure_hash_changes_are_rejected(setup):
    _, _, artifact = setup
    v.verify_bindings()
    artifact.write_text("changed overlay")
    with pytest.raises(ValueError, match="environment file changed"): v.verify_bindings()


def test_control_comparison_never_imports_predictions_and_is_hash_bound(setup):
    path = v.ROOT / r.CONFIG["v6_control_comparison_file"]
    put(path, {"control_count": 1})
    with pytest.raises(ValueError, match="comparison"): v.verify_bindings()


@pytest.mark.parametrize("relative", ["/etc/passwd", "../outside", "runtime/no-such-file"])
def test_environment_artifact_scope_rejects_unapproved_paths(setup, relative):
    with pytest.raises(ValueError, match="project file"): v.scoped_file(relative)


@pytest.mark.parametrize("action,model,judge", [("pilot", "qwen", "primary"), ("run", "qwen", "primary")])
def test_direct_qwen_stage_wrong_interpreter_fails_before_execution(setup, monkeypatch, action, model, judge):
    monkeypatch.setattr(v, "EXECUTE", lambda *args: pytest.fail("Wrong interpreter reached stage"))
    monkeypatch.setattr(sys, "prefix", str(v.ROOT / "runtime/app-env"))
    with pytest.raises(RuntimeError, match="judge-fast-env"): v.execute(action, model, judge)


@pytest.mark.parametrize("action,model,judge", [("pilot", "phi", "primary"), ("run", "qwen", "secondary"), ("start", "qwen", "primary")])
def test_phi_and_scheduler_remain_on_app_environment(setup, monkeypatch, action, model, judge):
    monkeypatch.setattr(sys, "prefix", str(v.ROOT / "runtime/app-env"))
    monkeypatch.setattr(v, "EXECUTE", lambda *args: args)
    assert v.execute(action, model, judge) == (action, model, judge, "all")


def test_fast_environment_stage_uses_frozen_v6_executor(setup, monkeypatch):
    monkeypatch.setattr(sys, "prefix", str(v.FAST.parent.parent))
    monkeypatch.setattr(v, "EXECUTE", lambda *args: args)
    assert v.execute("pilot", "qwen", "primary", "legacy-controls") == ("pilot", "qwen", "primary", "legacy-controls")


def test_failed_prepare_restores_original_freeze_hook(setup, monkeypatch):
    origin = r.OUT / "assessment_precedence_v5/validation/fresh"
    put(origin / "cases.jsonl", {}); put(origin / "manifest.json", {})
    monkeypatch.setattr(r.shared, "import_frozen", lambda *a: SimpleNamespace(generation_protocol=lambda: {"thinking_budget": 1024}))
    def failed(): raise ValueError("fixture error")
    monkeypatch.setattr(r, "SHARED_PREPARE", failed)
    before = r.ORIGINAL_FREEZE
    with pytest.raises(ValueError, match="fixture error"): v.prepare()
    assert r.ORIGINAL_FREEZE is before


@pytest.mark.parametrize("judge,model", [("primary", "qwen"), ("secondary", "phi")])
def test_frozen_full_handoff_survives_nested_configuration(setup, monkeypatch, judge, model):
    seen = {}
    def gate():
        r.shared.configure()
        assert r.engine.command is v.command and r.MODERN == v.FAST
        return {"accepted": True}
    def backend(actual):
        class Judge:
            def __init__(self, *a): seen["actual_model"] = actual
        return Judge
    def inner_prepare():
        r.shared.configure()
        assert r.engine.command is v.command
    monkeypatch.setattr(r, "check_gate", gate)
    monkeypatch.setattr(r, "backend", backend)
    monkeypatch.setattr(r, "prepare", inner_prepare)
    monkeypatch.setattr(r.utilities, "verify_weights", lambda path: seen.update(primary_weights=path))
    def full(arguments):
        selected = r.base.GPUJudge
        r.base.prepare()
        assert r.base.GPUJudge is selected
        r.base.GPUJudge({}, 36)
        r.base.download_primary()
        seen["secondary_weights"] = arguments.secondary_weights
    monkeypatch.setattr(r.base, "run_assessment", full)
    r.run_full(judge)
    assert seen == {"actual_model": model, "primary_weights": r.QWEN_WEIGHTS, "secondary_weights": r.PHI_WEIGHTS}
