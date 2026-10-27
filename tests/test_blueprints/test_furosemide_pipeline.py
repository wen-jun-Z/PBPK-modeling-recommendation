"""The independent evaluator must not run after failed or changed predictions."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_pipeline(monkeypatch, tmp_path, return_codes, mutate_after_methods=False):
    spec = importlib.util.spec_from_file_location(
        "furosemide_pipeline_under_test", ROOT / "scripts/run_furosemide_validation_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for name in ("DEPLOYMENT", "STATUS", "SUITE", "BASE", "PREDICTIONS", "DESIGN", "INPUT",
                 "PROPOSALS", "MANIFEST", "PYTHON"):
        monkeypatch.setattr(module, name, tmp_path / getattr(module, name).relative_to(ROOT))
    module.DEPLOYMENT.mkdir(parents=True)
    module.PROPOSALS.mkdir(parents=True)
    (module.PROPOSALS / "progress.json").write_text(json.dumps({"state": "completed", "completed": 3}))
    for path in (module.DESIGN, module.INPUT, module.MANIFEST, module.PROPOSALS / "manifest.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for name in ("run_blueprint_reviewed_suite.py", "predict_furosemide_external_study.py",
                 "score_blueprint_locked_endpoints.py"):
        path = tmp_path / "scripts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen\n")
    calls = []

    class Child:
        def __init__(self, command, **_):
            calls.append(command)
            self.pid = 10000 + len(calls)
            self.returncode = return_codes[len(calls) - 1]

        def poll(self):
            if mutate_after_methods and len(calls) == 1:
                module.DESIGN.write_text('{"changed":true}')
            return self.returncode

        def wait(self, **_):
            return self.returncode

    class Scope:
        def state(self, pid):
            return {"pid": pid, "start_ticks": "1"}

        def send(self, *_):
            return True

    monkeypatch.setattr(module.subprocess, "Popen", Child)
    monkeypatch.setattr(module, "ProcessScope", Scope)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    return module, calls


@pytest.mark.parametrize("return_codes,expected_calls", [([1], 1), ([0, 1], 2)])
def test_failed_stage_prevents_evaluation(monkeypatch, tmp_path, return_codes, expected_calls):
    module, calls = load_pipeline(monkeypatch, tmp_path, return_codes)
    with pytest.raises(RuntimeError, match="exited with code"):
        module.main()
    assert len(calls) == expected_calls
    assert not any("score_blueprint_locked_endpoints.py" in c[1] for c in calls)
    assert json.loads(module.STATUS.read_text())["status"] == "stopped"


def test_metadata_change_between_stages_stops_before_predictions(monkeypatch, tmp_path):
    module, calls = load_pipeline(monkeypatch, tmp_path, [0], mutate_after_methods=True)
    with pytest.raises(RuntimeError, match="input changed"):
        module.main()
    assert len(calls) == 1


def test_successful_stages_run_in_order_and_preserve_registration(monkeypatch, tmp_path):
    module, calls = load_pipeline(monkeypatch, tmp_path, [0, 0, 0])
    module.main()
    state = json.loads(module.STATUS.read_text())
    assert state["status"] == "completed"
    assert [s["name"] for s in state["steps"]] == ["methods", "predictions", "evaluation"]
    assert all(s["status"] == "completed" for s in state["steps"])
    assert len(calls) == 3
    assert state["external_observations_available_to_optimizer"] is False
    with pytest.raises(ValueError, match="existing registered pipeline"):
        module.main()
