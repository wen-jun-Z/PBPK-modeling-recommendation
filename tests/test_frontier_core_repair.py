"""Policy/provenance tests for the isolated repaired scoring run (no GPU)."""
from copy import deepcopy
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
runner = importlib.import_module("run_frontier_core_repair")


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(runner.base, "file_sha", lambda path: "frozen")
    monkeypatch.setattr(runner, "validation_error", lambda output, packet, queries: None if output == "valid" else "bad:"+output)
    case = {"case_id": "source-case", "kind": "real_source_pair", "packet": {}, "query": {"task_type": "ddi_exposure_prediction"}}
    old = {"output": "initial", "final_output": "ordinary", "raw_validation_error": "bad:initial",
           "attempts": [{"output": "initial", "batch_metrics": {"wall_seconds": 1}},
                        {"output": "ordinary", "batch_metrics": {"wall_seconds": 2}}]}
    return case, old


def test_corrected_initial_wins_without_rewriting_parent_history(policy):
    case, old = policy
    old["output"] = old["attempts"][0]["output"] = "valid"
    original = deepcopy(old)
    attempts, request = runner.adopted_attempts("qwen", "legacy", case, old, {"ordinary": {}, "focused": {}})
    assert request is None and [x["output"] for x in attempts] == ["valid"]
    assert old == original


def test_identical_feedback_reuses_ordinary_then_requests_focused(policy):
    case, old = policy
    attempts, request = runner.adopted_attempts("qwen", "legacy", case, old, {"ordinary": {}, "focused": {}})
    assert [x["output"] for x in attempts] == ["initial", "ordinary"]
    assert request["repair_kind"] == "focused" and request["prior_output"] == "ordinary"
    assert attempts[1]["generation_origin"] == "v8_ordinary_identical_feedback"


def test_changed_feedback_requires_new_ordinary_even_if_old_repair_was_valid(policy):
    case, old = policy
    old["raw_validation_error"] = "different historical validator failure"
    old["attempts"][1]["output"] = old["final_output"] = "valid"
    attempts, request = runner.adopted_attempts("qwen", "legacy", case, old, {"ordinary": {}, "focused": {}})
    assert len(attempts) == 1 and request["repair_kind"] == "ordinary"
    assert request["feedback"] == "bad:initial"


def test_controls_keep_initial_and_never_repair(policy):
    case, old = policy
    case["kind"] = "synthetic_control"
    attempts, request = runner.adopted_attempts("phi", "fresh", case, old, {"ordinary": {}, "focused": {}})
    assert request is None and len(attempts) == 1 and attempts[0]["output"] == "initial"


def test_focused_generation_is_journal_bound_and_stops_after_third_invalid(policy):
    case, old = policy
    _, request = runner.adopted_attempts("qwen", "legacy", case, old, {"ordinary": {}, "focused": {}})
    journal = {k:v for k,v in request.items() if k != "prior_output"}
    journal.update(output="still invalid", metrics={"wall_seconds": 3})
    journals = {"ordinary": {}, "focused": {case["case_id"]: journal}}
    attempts, request = runner.adopted_attempts("qwen", "legacy", case, old, journals)
    assert request is None and len(attempts) == 3 and attempts[-1]["output"] == "still invalid"
    journal["feedback"] = "silently changed"
    with pytest.raises(ValueError, match="exact ordered-policy feedback"):
        runner.adopted_attempts("qwen", "legacy", case, old, journals)


def test_new_ordinary_valid_prevents_focused_generation(policy):
    case, old = policy
    old["raw_validation_error"] = "different"
    _, request = runner.adopted_attempts("qwen", "legacy", case, old, {"ordinary": {}, "focused": {}})
    journal = {k:v for k,v in request.items() if k != "prior_output"}
    journal.update(output="valid", metrics={})
    attempts, request = runner.adopted_attempts("qwen", "legacy", case, old, {"ordinary": {case["case_id"]: journal}, "focused": {}})
    assert request is None and len(attempts) == 2 and attempts[-1]["output"] == "valid"


def test_production_extra_attempt_is_real_and_all_three_outputs_are_journaled(monkeypatch, policy):
    writes, calls = [], []
    monkeypatch.setattr(runner.base, "append_rows", lambda path, rows: writes.append((path, deepcopy(rows))))
    class FakeBackend:
        def generate(self, tasks, max_new_tokens=96, repairs=None):
            calls.append((max_new_tokens, deepcopy(repairs)))
            return (["ordinary invalid"] if len(calls) == 1 else ["valid"]), {"wall_seconds": len(calls)}
    judge = runner.repaired_backend(FakeBackend)()
    packet, query = {"packet_sha256": "packet"}, {"query_id": "query", "task_type": "ddi_exposure_prediction"}
    output, metrics = judge.generate([(packet, [query])], max_new_tokens=160, repairs=[{"output": "initial", "error": "bad"}])
    assert output == ["valid"] and len(calls) == 2
    assert calls[1][1][0]["output"] == "ordinary invalid"
    assert [a["output"] for a in metrics["actual_repair_attempts"]] == ["ordinary invalid", "valid"]
    assert writes[0][1][0]["initial_output"] == "initial"
    captured = []
    monkeypatch.setattr(runner, "_append", lambda path, rows: captured.extend(rows))
    runner.append_with_explicit_attempts(runner.ASSESS / "primary_generation_journal.jsonl",
        [{"attempts": [{"output": "initial", "batch_metrics": {}}, {"output": output[0], "batch_metrics": metrics}]}])
    assert [a["output"] for a in captured[0]["attempts"]] == ["initial", "ordinary invalid", "valid"]


def test_production_valid_ordinary_does_not_call_extra(monkeypatch, policy):
    calls = []
    class FakeBackend:
        def generate(self, tasks, max_new_tokens=96, repairs=None):
            calls.append(1)
            return ["valid"], {}
    judge = runner.repaired_backend(FakeBackend)()
    assert judge.generate([({}, [{}])], repairs=[{"output": "bad", "error": "bad"}])[0] == ["valid"]
    assert len(calls) == 1


def test_acceptance_rejects_missing_review(monkeypatch):
    monkeypatch.setattr(runner.base, "read", lambda path, default=None: {})
    with pytest.raises(RuntimeError, match="source-reviewed"):
        runner.verify_acceptance()
