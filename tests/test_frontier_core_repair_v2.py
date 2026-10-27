"""Actual-call reuse must preserve complete inputs and explicit provenance."""
import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
runner = importlib.import_module("run_frontier_core_repair_v2")


@pytest.fixture
def call_request():
    return {"case_id": "case", "model": "qwen", "cohort": "legacy", "repair_kind": "focused",
            "prior_output_sha256": "sha", "prior_output": "old", "feedback": "exact feedback",
            "pilot_protocol_sha256": "new-profile"}


def test_exact_call_can_reuse_despite_new_envelope_protocol(call_request):
    old = {**call_request, "pilot_protocol_sha256": "old-profile", "output": "actual output", "metrics": {}}
    assert runner.reusable_row(old, call_request)


@pytest.mark.parametrize("key", ["case_id", "model", "cohort", "repair_kind", "prior_output_sha256", "feedback"])
def test_every_changed_message_identity_component_prevents_reuse(call_request, key):
    old = {**call_request, "output": "actual output", "metrics": {}, key: "different"}
    assert not runner.reusable_row(old, call_request)


def test_corrected_initial_does_not_import_unnecessary_historical_repairs(monkeypatch):
    monkeypatch.setattr(runner, "equal_generation_contract", lambda: None)
    monkeypatch.setattr(runner, "_replay", lambda model: [])
    monkeypatch.setattr(runner, "import_actual_generation", lambda *args: pytest.fail("No repair should be reused"))
    assert runner.replay("qwen") == []


def test_staged_replay_reuses_only_when_pending_then_finishes(monkeypatch):
    request = {"model": "qwen"}
    states = [[("legacy", {"case_id": "x"}, {}, request)], []]
    called = []
    monkeypatch.setattr(runner, "equal_generation_contract", lambda: None)
    monkeypatch.setattr(runner, "_replay", lambda model: states.pop(0))
    monkeypatch.setattr(runner, "import_actual_generation", lambda *args: called.append(args) or True)
    assert runner.replay("qwen") == [] and len(called) == 1


def test_old_actual_generation_is_never_labeled_new_gpu(monkeypatch):
    origin = {"journal_path": "source", "journal_sha256": "digest"}
    attempts = [{"output": "initial", "generation_origin": "v8_initial"},
                {"output": "result", "generation_origin": "new_focused"}]
    monkeypatch.setattr(runner, "_adopted", lambda *args: (copy.deepcopy(attempts), None))
    checked = []
    monkeypatch.setattr(runner, "verify_reused_record", lambda row: checked.append(row))
    record = {"generation_origin": "reused_actual_v1_generation", "reused_from": origin}
    derived, pending = runner.adopted_attempts("qwen", "legacy", {"case_id": "x"}, {},
                                             {"focused": {"x": record}})
    assert pending is None and len(checked) == 1
    assert derived[1]["generation_origin"] == "reused_actual_v1_generation"
    assert derived[1]["reused_from"] == origin


def test_configure_scope_does_not_leak_v1_default_directories(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "_scope", lambda *args: calls.append(args))
    runner.configure_scope()
    assert calls == [(runner.QUAL, runner.QUAL_RUN)]


def test_changed_decoding_contract_refuses_call_reuse(monkeypatch):
    parent = {"generation": {"max_new_tokens": 160}}
    current = {"generation": {"max_new_tokens": 200}}
    monkeypatch.setattr(runner.base, "read", lambda path: parent if path.parent == runner.PARENT_QUAL else current)
    with pytest.raises(ValueError, match="generation budgets"):
        runner.equal_generation_contract()


def test_production_returns_exact_frozen_protocol_after_hook(monkeypatch):
    stored = {}
    def freeze(path, value):
        stored[path] = copy.deepcopy(value)
        return value
    monkeypatch.setattr(runner.v1.core, "freeze", freeze)
    monkeypatch.setattr(runner.base, "file_sha", lambda path: "source-hash")
    monkeypatch.setattr(runner.base, "read", lambda path: copy.deepcopy(stored[path]))
    def original_prepare():
        protocol = {"version": "old-version", "input_sha256": {
            "assessment_repaired/sources.jsonl": "source-hash",
            "assessment_repaired/source_supplements/sources.jsonl": "source-hash"}}
        stale_return = copy.deepcopy(protocol)
        runner.v1.core.freeze(runner.ASSESS / "judge_protocol.json", protocol)
        return ["query"], ["candidate"], {"packet": 1}, stale_return
    monkeypatch.setattr(runner, "_prepare_core", original_prepare)
    result = runner.prepare_core()
    assert result[3] == stored[runner.ASSESS / "judge_protocol.json"]
    assert result[3]["version"] == "pbpk-core145-repaired-production-v2"
    assert runner.ASSESS.name + "/sources.jsonl" in result[3]["input_sha256"]
    assert "assessment_repaired/sources.jsonl" not in result[3]["input_sha256"]
    assert runner.v1.core.freeze is freeze
