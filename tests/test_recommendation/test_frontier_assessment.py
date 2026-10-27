"""Offline safeguards for automatic source judgments and exact-content reuse."""
from __future__ import annotations

from contextlib import nullcontext
from collections import Counter
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_frontier_assessment as assessment


def source(candidate="c0", text="The PBPK model supports oral exposure prediction.", *, path="model/report.md", family="family"):
    return {
        "candidate_id": candidate, "family_name": family, "compounds": ["drug"],
        "source_artifact": {"verified": True},
        "documents": [{"path": path, "sha256": assessment.sha(text.encode()),
            "version_specific_validation_verified": False,
            "segments": [{"excerpt": text, "excerpt_sha256": assessment.sha(text.encode()),
                "evidence_kind": "model_description", "evidence_id": f"source:{candidate}",
                "location": {"char_start": 0, "char_end": len(text)}}]}],
    }


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def scoped_fixture(tmp_path, monkeypatch):
    root, run, out = tmp_path, tmp_path / "runtime", tmp_path / "frontier"
    target = out / "assessment"
    run.mkdir(parents=True)
    target.mkdir(parents=True)
    monkeypatch.setattr(assessment, "ROOT", root)
    monkeypatch.setattr(assessment, "RUN", run)
    monkeypatch.setattr(assessment, "OUT", out)
    monkeypatch.setattr(assessment, "ASSESS", target)
    return root, run, out, target


def test_packet_preserves_exact_quotes_hashes_and_rebinds_provenance():
    first = assessment.build_packet(source())
    second = assessment.build_packet(source("c1", path="other/report.md"))
    assert first["packet_sha256"] == second["packet_sha256"]
    assert first["refs"][0]["source_id"] != second["refs"][0]["source_id"]
    assert first["refs"][0]["quote"] == first["body"]["excerpts"][0]["text"]
    assert assessment.sha(first["refs"][0]["quote"].encode()) == first["refs"][0]["excerpt_sha256"]
    damaged = source()
    damaged["documents"][0]["segments"][0]["excerpt"] += " fabricated"
    with pytest.raises(ValueError, match="hash mismatch"):
        assessment.build_packet(damaged)


def test_cache_is_exact_content_and_complete_request_not_family():
    packet = assessment.build_packet(source())
    query = {"query_id": "q0", "text": "Predict adult oral exposure"}
    identical = {**query, "query_id": "q1"}
    assert assessment.task_key(packet, query) == assessment.task_key(packet, identical)
    assert assessment.task_key(packet, query) != assessment.task_key(packet, {**query, "text": "Predict pediatric oral exposure"})
    different_material = assessment.build_packet(source("c1", "This model excludes oral exposure prediction."))
    assert assessment.task_key(packet, query) != assessment.task_key(different_material, query)


def test_packet_budget_omits_complete_excerpt_without_slicing(monkeypatch):
    text = "one complete observation"
    monkeypatch.setattr(assessment, "PACKET_CHARS", len(text) - 1)
    packet = assessment.build_packet(source(text=text))
    assert packet["refs"] == [] and packet["body"]["excerpts"] == []
    assert packet["omitted_unique_segments"] == 1


@pytest.mark.parametrize("answer", [
    [{"i": 1, "g": 3, "a": "Y", "e": []}],
    [{"i": 1, "g": 3, "a": "Y", "e": [2]}],
    [{"i": 1, "g": 3, "a": "Y", "e": [1, 1]}],
    [{"i": 1, "g": True, "a": "Y", "e": [1]}],
    [{"i": 1, "g": 3, "a": "Y", "e": [True]}],
    [{"i": 2, "g": 3, "a": "Y", "e": [1]}],
    [{"i": 1, "g": 3, "a": "Y", "e": [1], "confidence": 1}],
])
def test_parser_rejects_missing_false_citations_and_schema(answer):
    with pytest.raises(ValueError):
        assessment.parse_answers(json.dumps(answer), [1], [{"source_id": "real"}])


def test_parser_rejects_duplicate_request_ids():
    answer = [{"i": 1, "g": 3, "a": "Y", "e": [1]}] * 2
    with pytest.raises(ValueError, match="repeated request"):
        assessment.parse_answers(json.dumps(answer), [1, 2], [{}])


def test_unknown_stays_unknown_and_grade_is_not_applicability():
    raw = [{"i": 1, "g": None, "a": "U", "e": []},
           {"i": 2, "g": 1, "a": "N", "e": [1]},
           {"i": 3, "g": 2, "a": "U", "e": [1]}]
    parsed = assessment.parse_answers(json.dumps(raw), [1, 2, 3], [{}])
    assert parsed[1]["grade"] is None and parsed[1]["documented_applicability"] is None
    assert parsed[1]["status"] == "insufficient_evidence"
    assert parsed[2]["grade"] == 1 and parsed[2]["documented_applicability"] is False
    assert parsed[3]["grade"] == 2 and parsed[3]["documented_applicability"] is None


def test_newline_recovery_preserves_committed_prefix_and_archives_tail(tmp_path):
    path = tmp_path / "journal.jsonl"
    assessment.append_rows(path, [{"key": 1}, {"key": 2}])
    committed = path.read_bytes()
    tail = b'{"key": "unfinished'
    with path.open("ab") as handle:
        handle.write(tail)
    with pytest.raises(ValueError, match="Uncommitted"):
        assessment.load_committed(path)
    assert assessment.load_committed(path, repair_tail=True) == [{"key": 1}, {"key": 2}]
    assert path.read_bytes() == committed
    archive = path.with_suffix(".jsonl.interrupted")
    assert archive.read_bytes() == tail + b"\n"
    assessment.load_committed(path, repair_tail=True)
    assert archive.read_bytes() == tail + b"\n"


def test_malformed_committed_record_is_not_silently_repaired(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b'{"key": 1}\nnot-json\n')
    with pytest.raises(json.JSONDecodeError):
        assessment.load_committed(path, repair_tail=True)


@pytest.mark.parametrize("duplicate", ["queries", "candidates", "sources"])
def test_prepare_rejects_duplicate_dataset_ids(tmp_path, monkeypatch, duplicate):
    _, _, out, target = scoped_fixture(tmp_path, monkeypatch)
    queries = [{"query_id": f"q{i}", "text": str(i)} for i in range(772)]
    candidates = [{"candidate_id": f"c{i}"} for i in range(280)]
    sources = [source(f"c{i}") for i in range(280)]
    if duplicate == "queries":
        queries[-1] = copy.deepcopy(queries[0])
    elif duplicate == "candidates":
        candidates[-1] = copy.deepcopy(candidates[0])
        sources[-1] = copy.deepcopy(sources[0])
    else:
        sources.append(copy.deepcopy(sources[0]))
    jsonl(out / "inputs/queries.jsonl", queries)
    jsonl(out / "inputs/candidates.jsonl", candidates)
    jsonl(target / "sources.jsonl", sources)
    with pytest.raises(ValueError, match="[Dd]uplicate|unique"):
        assessment.prepare()


def fake_runner(tmp_path, monkeypatch, *, identical_queries=False, malformed=False, fewer_outputs=False):
    _, run, out, target = scoped_fixture(tmp_path, monkeypatch)
    queries = [{"query_id": "q0", "text": "Predict exposure"},
               {"query_id": "q1", "text": "Predict exposure" if identical_queries else "Predict DDI"}]
    candidates = [{"candidate_id": "c0"}, {"candidate_id": "c1"}]
    packets = {c["candidate_id"]: assessment.build_packet(source(c["candidate_id"], path=f"{c['candidate_id']}/report.md")) for c in candidates}
    protocol = {"fixture": "frozen"}
    (target / "judge_protocol.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(assessment, "prepare", lambda: (queries, candidates, packets, protocol))
    monkeypatch.setattr(assessment, "download_primary", lambda: {"repo": "fixture", "revision": "fixed"})
    monkeypatch.setattr(assessment, "identity", lambda: {"pid": 123, "start_ticks": "fixture"})
    monkeypatch.setattr(assessment.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**40))
    calls = []
    class FakeJudge:
        torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=MemoryError, empty_cache=lambda: None))
        def __init__(self, *args):
            self.memory_cap_gib = args[1]
            self.prompt_batch_limit = 8
            self.batch_token_budget = 8000
            self.capacity_upgrade = None
        def maybe_expand_capacity(self):
            return None
        def select_batch_size(self, tasks, maximum):
            return min(len(tasks), maximum)
        def generate(self, tasks, max_new_tokens=512, repairs=None):
            calls.append([(p["packet_sha256"], [q["query_id"] for q in qs]) for p, qs in tasks])
            outputs = []
            for packet, qs in tasks:
                outputs.append("invalid" if malformed else json.dumps([
                    {"i": i + 1, "g": 2, "a": "Y", "e": [1]} for i in range(len(qs))]))
            if fewer_outputs:
                outputs = outputs[:-1]
            return outputs, {"wall_seconds": .1}
    monkeypatch.setattr(assessment, "GPUJudge", FakeJudge)
    args = SimpleNamespace(judge="primary", memory_gib=10, batch_size=4, max_batches=0)
    return args, target, run, calls


def test_exact_cache_fans_out_with_each_candidates_real_source_refs(tmp_path, monkeypatch):
    args, target, run, calls = fake_runner(tmp_path, monkeypatch, identical_queries=True)
    assessment.run_assessment(args)
    cache = assessment.load_committed(target / "primary_judgment_cache.jsonl")
    records = assessment.load_committed(target / "judgments.jsonl")
    assert len(cache) == 1
    assert len(records) == 4
    assert len({(r["query_id"], r["candidate_id"]) for r in records}) == 4
    assert all(r["support"][0]["source_id"] == f"source:{r['candidate_id']}" for r in records)
    assert sum(len(qids) for call in calls for _, qids in call) == 1
    before = (target / "judgments.jsonl").read_bytes()
    assessment.run_assessment(args)
    assert (target / "judgments.jsonl").read_bytes() == before


def test_invalid_generation_has_bounded_retry_then_unknown_not_zero(tmp_path, monkeypatch):
    args, target, _, calls = fake_runner(tmp_path, monkeypatch, malformed=True)
    assessment.run_assessment(args)
    assert len(calls) == 1 + len(calls[0])  # One initial batch and one retry per prompt.
    assert Counter(qid for batch in calls for _, qids in batch for qid in qids) == {"q0": 2, "q1": 2}
    records = assessment.load_committed(target / "judgments.jsonl")
    assert len(records) == 4
    assert all(r["grade"] is None and r["documented_applicability"] is None and r["status"] == "invalid_output" and not r["support"] for r in records)


@pytest.mark.parametrize("fault", ["protocol", "extra_pair", "source_packet", "source_type", "quote", "source_id", "grade", "task_key", "generation_id"])
def test_resume_rejects_stale_or_out_of_scope_final_rows(tmp_path, monkeypatch, fault):
    args, target, _, _ = fake_runner(tmp_path, monkeypatch)
    assessment.run_assessment(args)
    records = assessment.load_committed(target / "judgments.jsonl")
    if fault == "protocol":
        records[0]["protocol_sha256"] = "old"
    elif fault == "extra_pair":
        records[0]["candidate_id"] = "not-in-pool"
    elif fault == "source_packet":
        records[0]["source_packet_sha256"] = "different-material"
    elif fault == "source_type":
        records[0]["label_source"] = "blinded_expert"
    elif fault in ("quote", "source_id"):
        records[0]["support"][0][fault] = "fabricated"
    elif fault == "grade":
        records[0]["grade"] = 3
    else:
        records[0][fault] = "different"
    jsonl(target / "judgments.jsonl", records)
    with pytest.raises(ValueError):
        assessment.run_assessment(args)


def test_short_backend_output_cannot_mark_missing_pairs_complete(tmp_path, monkeypatch):
    args, target, run, _ = fake_runner(tmp_path, monkeypatch, fewer_outputs=True)
    with pytest.raises((ValueError, RuntimeError)):
        assessment.run_assessment(args)
    state = json.loads((run / "status.json").read_text())
    assert state["state"] != "complete"


@pytest.mark.parametrize("fault", ["missing", "foreign_key", "reference", "grade", "status"])
def test_resume_rejects_corrupt_cache_without_overwriting_results(tmp_path, monkeypatch, fault):
    args, target, _, _ = fake_runner(tmp_path, monkeypatch)
    assessment.run_assessment(args)
    path = target / "primary_judgment_cache.jsonl"
    records = assessment.load_committed(path)
    if fault == "missing":
        records.pop()
    elif fault == "foreign_key":
        records[0]["task_key"] = "foreign"
    elif fault == "reference":
        records[0]["evidence_numbers"] = [100]
    elif fault == "grade":
        records[0]["grade"] = True
    else:
        records[0]["status"] = "invalid_output"
    jsonl(path, records)
    before = (target / "judgments.jsonl").read_bytes()
    with pytest.raises(ValueError):
        assessment.run_assessment(args)
    assert (target / "judgments.jsonl").read_bytes() == before


def agreement_fixture(tmp_path, monkeypatch):
    _, _, out, target = scoped_fixture(tmp_path, monkeypatch)
    queries = [{"query_id": "q0"}, {"query_id": "q1"}]
    candidates = [{"candidate_id": "c0"}, {"candidate_id": "c1"}]
    jsonl(out / "inputs/queries.jsonl", queries)
    jsonl(out / "inputs/candidates.jsonl", candidates)
    monkeypatch.setattr(assessment, "secondary_selected", lambda q, c: q == "q0")
    protocol_path = target / "judge_protocol.json"
    protocol_path.write_text('{"fixture":true}')
    primary = [{"query_id": q["query_id"], "candidate_id": c["candidate_id"],
                "grade": 2, "documented_applicability": True,
                "judge": "primary", "label_source": "llm_source_grounded",
                "protocol_sha256": assessment.file_sha(protocol_path)} for q in queries for c in candidates]
    secondary = [dict(row, judge="secondary") for row in primary[:2]]
    secondary[1].update(grade=None, documented_applicability=None)
    jsonl(target / "judgments.jsonl", primary)
    jsonl(target / "secondary_judgments.jsonl", secondary)
    return target, primary, secondary


def test_agreement_has_exact_sample_coverage_and_unknown_denominator(tmp_path, monkeypatch):
    target, primary, secondary = agreement_fixture(tmp_path, monkeypatch)
    original = (target / "judgments.jsonl").read_bytes()
    result = assessment.judge_agreement()
    assert result["sampled_pairs"] == result["expected_sample_pairs"] == 2
    assert result["primary_pair_count"] == 4
    assert result["grade_known_both"] == result["applicability_known_both"] == 1
    assert result["grade_exact_agreement"] == 1
    assert result["primary_labels_replaced"] is False
    assert (target / "judgments.jsonl").read_bytes() == original


@pytest.mark.parametrize("fault", ["primary_duplicate", "secondary_duplicate", "primary_missing", "secondary_missing", "secondary_foreign", "protocol", "tail"])
def test_agreement_rejects_incomplete_or_wrong_sample(tmp_path, monkeypatch, fault):
    target, primary, secondary = agreement_fixture(tmp_path, monkeypatch)
    if fault == "primary_duplicate":
        primary.append(primary[0])
    elif fault == "secondary_duplicate":
        secondary.append(secondary[0])
    elif fault == "primary_missing":
        primary.pop()
    elif fault == "secondary_missing":
        secondary.pop()
    elif fault == "secondary_foreign":
        secondary[0]["query_id"] = "q1"
    elif fault == "protocol":
        secondary[0]["protocol_sha256"] = "wrong"
    jsonl(target / "judgments.jsonl", primary)
    jsonl(target / "secondary_judgments.jsonl", secondary)
    if fault == "tail":
        with (target / "secondary_judgments.jsonl").open("a") as handle:
            handle.write("unfinished")
    with pytest.raises(ValueError):
        assessment.judge_agreement()


def pipeline_fixture(tmp_path, monkeypatch, mode="complete"):
    root, run, out, target = scoped_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(assessment, "prepare", lambda: None)
    monkeypatch.setattr(assessment, "identity", lambda: {"pid": 123, "start_ticks": "fixture"})
    monkeypatch.setattr(assessment, "environment", lambda: {})
    monkeypatch.setattr(assessment, "alive", lambda saved: False)
    monkeypatch.setattr(assessment.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**40))
    events, children = [], []
    clock = {"seconds": 0}
    def sleep(seconds):
        clock["seconds"] += seconds
        if mode == "stop":
            (run / "stop.requested").touch()
        if mode == "interrupt":
            raise KeyboardInterrupt("simulated shutdown")
    monkeypatch.setattr(assessment, "time", SimpleNamespace(monotonic=lambda: clock["seconds"], sleep=sleep))
    def evaluate(*args):
        events.append("evaluate")
        return {"state": "complete" if events.count("evaluate") == 2 else "partial"}
    monkeypatch.setitem(sys.modules, "evaluate_frontier_automatic", SimpleNamespace(evaluate_run=evaluate))
    monkeypatch.setitem(sys.modules, "run_frontier_comparison", SimpleNamespace(
        METHODS={"ranker": {}}, RUN=run / "rankings", CACHE=root / "cache", report=lambda: events.append("rank_report")))
    monkeypatch.setattr(assessment, "judge_agreement", lambda: (events.append("agreement") or {"sampled_pairs": 2}))
    monkeypatch.setattr(assessment, "write_readme_results", lambda _: events.append("readme"), raising=False)
    (out / "results/ranker").mkdir(parents=True)
    (out / "results/ranker/status.json").write_text('{"state":"complete"}')
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 1000 + len(children)
            self.name = command[command.index("--judge") + 1]
            self.terminated = self.killed = 0
            self.returncode = 0 if mode in ("complete", "wrong_pid", "failed") else None
            children.append(self)
            events.append(self.name)
            status = {"state": "complete" if self.returncode == 0 else "judging", "pid": self.pid, "judge": self.name,
                      "committed_pairs": 0}
            if mode in ("stale_status", "wrong_pid"):
                status.update(state="complete", pid=999)
            if mode == "failed":
                self.returncode = 1
                status["state"] = "failed"
            (run / "status.json").write_text("malformed-json" if mode == "read_error" else json.dumps(status))
        def poll(self):
            return self.returncode
        def terminate(self):
            self.terminated += 1
            self.returncode = -15
        def wait(self, timeout):
            return self.returncode
        def kill(self):
            self.killed += 1
            self.returncode = -9
    monkeypatch.setattr(assessment.subprocess, "Popen", Child)
    return SimpleNamespace(batch_size=4, memory_gib=10), run, events, children


def test_pipeline_orders_full_primary_secondary_wait_and_final_report(tmp_path, monkeypatch):
    args, run, events, children = pipeline_fixture(tmp_path, monkeypatch)
    assessment.pipeline(args)
    assert events == ["primary", "evaluate", "readme", "secondary", "agreement", "rank_report", "evaluate", "readme"]
    assert json.loads((run / "pipeline_status.json").read_text())["state"] == "complete"
    assert all(child.terminated == 0 for child in children)


@pytest.mark.parametrize("mode", ["stalled", "stale_status", "stop"])
def test_pipeline_watchdog_stops_only_its_child_and_never_restarts(tmp_path, monkeypatch, mode):
    args, run, events, children = pipeline_fixture(tmp_path, monkeypatch, mode)
    assessment.pipeline(args)
    result = json.loads((run / "pipeline_status.json").read_text())
    assert result["state"] == "stopped_on_guard"
    assert result["guard_reason"] == ("UserStopRequested" if mode == "stop" else "NoProgress20Minutes")
    assert len(children) == 1 and children[0].terminated == 1
    assert "secondary" not in events and "evaluate" not in events


@pytest.mark.parametrize("mode", ["read_error", "interrupt"])
def test_pipeline_exception_or_interrupt_cleans_own_live_child(tmp_path, monkeypatch, mode):
    args, run, events, children = pipeline_fixture(tmp_path, monkeypatch, mode)
    with pytest.raises(SystemExit):
        assessment.pipeline(args)
    assert len(children) == 1 and children[0].terminated == 1
    assert json.loads((run / "pipeline_status.json").read_text())["state"] == ("interrupted" if mode == "interrupt" else "failed")


@pytest.mark.parametrize("mode", ["wrong_pid", "failed"])
def test_pipeline_worker_failure_or_wrong_completion_identity_does_not_advance(tmp_path, monkeypatch, mode):
    args, run, events, children = pipeline_fixture(tmp_path, monkeypatch, mode)
    assessment.pipeline(args)
    assert json.loads((run / "pipeline_status.json").read_text())["state"] == "failed"
    assert events == ["primary"]


def test_stop_child_handles_exit_and_bounded_terminate_kill():
    class Child:
        returncode = None
        terminated = killed = waits = 0
        def poll(self):
            return self.returncode
        def terminate(self):
            self.terminated += 1
        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise assessment.subprocess.TimeoutExpired("own_child", timeout)
            self.returncode = -9
        def kill(self):
            self.killed += 1
    child = Child()
    assessment.stop_child(child)
    assert (child.terminated, child.killed, child.waits) == (1, 1, 2)
    assessment.stop_child(child)
    assert (child.terminated, child.killed, child.waits) == (1, 1, 2)


def test_gpu_generate_preserves_official_multiple_eos_without_using_a_gpu():
    """The tokenizer's single EOS must not override Phi's full stop-token list."""
    official_eos = [1001, 1002, 1003]
    captured = {}
    class Tensor:
        def __init__(self, value):
            self.array = np.asarray(value)
            self.shape = self.array.shape
        def to(self, target):
            captured.setdefault("requested_devices", []).append(target)
            return self
        def __getitem__(self, indices):
            return Tensor(self.array[indices])
    class Tokenizer:
        eos_token_id = official_eos[0]
        pad_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return messages[-1]["content"]
        def __call__(self, texts, **kwargs):
            assert kwargs["padding"] is True and kwargs["truncation"] is False
            return {"input_ids": Tensor([[0, 10, 11], [12, 13, 14]]),
                    "attention_mask": Tensor([[0, 1, 1], [1, 1, 1]])}
        def batch_decode(self, outputs, **kwargs):
            assert outputs.shape == (2, 2)
            assert kwargs["skip_special_tokens"] is True
            return ["first", "second"]
    def generate(**kwargs):
        captured.update(kwargs)
        return Tensor([[0, 10, 11, 31, official_eos[1]], [12, 13, 14, 32, official_eos[2]]])
    judge = assessment.GPUJudge.__new__(assessment.GPUJudge)
    judge.tokenizer = Tokenizer()
    judge.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=official_eos), generate=generate)
    judge.torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(
        synchronize=lambda: None, max_memory_allocated=lambda _: 0, max_memory_reserved=lambda _: 0))
    callback = lambda batch_id, tokens: [official_eos[0]]
    def constrain(tasks):
        captured["constrained_tasks"] = tasks
        return callback
    judge.json_constraint = SimpleNamespace(for_tasks=constrain)
    judge.memory_cap_gib = 12.0
    judge.prompt_batch_limit = 8
    judge.batch_token_budget = 8000
    judge.capacity_upgrade = None
    packet = assessment.build_packet(source())
    tasks = [(packet, [{"text": "first request"}]), (packet, [{"text": "second request"}])]
    output, metrics = judge.generate(tasks, max_new_tokens=20)
    assert captured.get("eos_token_id", judge.model.generation_config.eos_token_id) == [1001, 1002, 1003]
    assert judge.model.generation_config.eos_token_id == [1001, 1002, 1003]
    assert captured["constrained_tasks"] is tasks
    assert captured["prefix_allowed_tokens_fn"] is callback
    assert output == ["first", "second"] and captured["do_sample"] is False
    assert metrics["prompt_batch_size"] == 2 and metrics["output_token_width"] == 2
    assert metrics["gpu_memory_cap_gib"] == 12.0 and metrics["batch_token_budget"] == 8000


def readme_result(*, state="partial", assessment_state="partial"):
    return {"state": state, "created_at": "fixed-time", "assessment": {
        "state": assessment_state, "processed_unique_pairs": 24 if assessment_state == "complete" else 8,
        "expected_pairs": 24, "grade_known_pairs": 7},
        "method_counts": {"included_complete_matrices": 1, "expected": 2},
        "methods": {"should_not_publish_partial_method": {"variants": {"original": {
            "overall": {"judged_only": {"metrics": {"ndcg@10": .987}}},
            "calibrated_applicability": {"macro_f1": .9}}}}}}


def test_readme_update_preserves_handwritten_content_outside_markers(tmp_path, monkeypatch):
    scoped_fixture(tmp_path, monkeypatch)
    path = tmp_path / "experiments/model_recommendation/README.md"
    path.parent.mkdir(parents=True)
    first = "<!-- frontier-automatic-results:start -->"
    last = "<!-- frontier-automatic-results:end -->"
    prefix = "# 手写核心说明\n\n原始实验成果必须保留。\n\n"
    suffix = "\n\n后续手写计划\n\n  保留原缩进与末尾空白  \n"
    path.write_text(prefix + first + "\nold generated data\n" + last + suffix)
    assessment.write_readme_results(readme_result())
    updated = path.read_text()
    assert updated.startswith(prefix + first) and updated.endswith(last + suffix)
    assert "old generated data" not in updated
    assert updated.count(first) == updated.count(last) == 1
    assessment.write_readme_results(readme_result())
    assert path.read_text() == updated


@pytest.mark.parametrize("assessment_state", ["partial", "complete"])
def test_partial_readme_does_not_publish_method_ranking(tmp_path, monkeypatch, assessment_state):
    scoped_fixture(tmp_path, monkeypatch)
    path = tmp_path / "experiments/model_recommendation/README.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Existing project prose\n")
    assessment.write_readme_results(readme_result(state="partial", assessment_state=assessment_state))
    value = path.read_text()
    assert "| 方法 |" not in value
    assert "should\\_not\\_publish\\_partial\\_method" not in value
    assert "0.987" not in value


@pytest.mark.parametrize("markers", ["<!-- frontier-automatic-results:start -->", "<!-- frontier-automatic-results:end --><!-- frontier-automatic-results:start -->"])
def test_bad_readme_markers_do_not_overwrite_manual_prose(tmp_path, monkeypatch, markers):
    scoped_fixture(tmp_path, monkeypatch)
    path = tmp_path / "experiments/model_recommendation/README.md"
    path.parent.mkdir(parents=True)
    path.write_text("handwritten\n" + markers + "\nretained")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Malformed generated README"):
        assessment.write_readme_results(readme_result())
    assert path.read_bytes() == original


def scope_answer(grade=3, applicable=True, evidence_numbers=None):
    return {"grade": grade, "documented_applicability": applicable,
            "evidence_numbers": [1] if evidence_numbers is None else evidence_numbers}


def test_scope_guard_rejects_navigation_links_as_special_population_support():
    refs = [{"quote": "## Midazolam model\n- [Pregnancy, human adult oral model](https://example.test/pregnancy)"}]
    query = {"features": {"populations": ["pregnancy"], "species": ["human"], "routes": ["oral"]}}
    with pytest.raises(ValueError, match="Unsupported explicit scope.*populations=pregnancy"):
        assessment.validate_explicit_scope(scope_answer(), query, refs)
    refs[0]["quote"] = '<a href="https://example.test">Pregnant human volunteers, oral dose</a>'
    with pytest.raises(ValueError, match="populations=pregnancy"):
        assessment.validate_explicit_scope(scope_answer(), query, refs)


@pytest.mark.parametrize("population,prose", [
    ("pediatric", "The study evaluated oral exposure in children."),
    ("pediatric", "The model was evaluated in paediatric patients."),
    ("adult", "The model describes adult volunteers."),
    ("geriatric", "Exposure was evaluated in older adults."),
    ("pregnancy", "The model was evaluated in pregnant women."),
    ("renal_impairment", "Exposure was assessed in patients with chronic kidney disease."),
    ("hepatic_impairment", "The evaluation included Child-Pugh B patients."),
])
def test_scope_guard_accepts_explicit_population_synonyms_without_assigning_grade(population, prose):
    answer = scope_answer()
    before = copy.deepcopy(answer)
    result = assessment.validate_explicit_scope(answer, {"features": {"populations": [population]}}, [{"quote": prose}])
    assert result["applied"] is True and result["semantic_entailment_verified"] is False
    assert answer == before


@pytest.mark.parametrize("route,prose", [
    ("intravenous", "The participants received an IV dose of 1 mg."),
    ("intravenous", "The study simulated i.v. dosing."),
    ("oral", "The study administered the drug p.o. to volunteers."),
    ("oral", "The participants received the dose by mouth."),
    ("subcutaneous", "The study simulated subcutaneous administration."),
    ("inhalation", "Participants received inhaled doses."),
])
def test_scope_guard_accepts_requested_route_synonyms(route, prose):
    result = assessment.validate_explicit_scope(scope_answer(), {"features": {"routes": [route]}}, [{"quote": prose}])
    assert result["checked_constraints"] == [{"dimension": "routes", "value": route}]


@pytest.mark.parametrize("species,prose", [
    ("human", "The model describes healthy volunteers."),
    ("human", "The model was evaluated in pregnant women."),
    ("human", "The evaluation used clinical observations."),
    ("mouse", "The model was evaluated using measured exposure in mice."),
    ("rat", "The model was evaluated using rat observations."),
    ("dog", "The model was evaluated in beagles."),
])
def test_scope_guard_accepts_species_synonyms(species, prose):
    assert assessment.validate_explicit_scope(scope_answer(), {"features": {"species": [species]}}, [{"quote": prose}])["applied"]


def test_scope_guard_checks_only_cited_body_not_uncited_text_paths_or_mechanism_mentions():
    refs = [{"quote": "The model is available at the repository.", "document_path": "pregnant_human_oral.md"},
            {"quote": "The model was evaluated in pregnant women receiving oral doses."}]
    query = {"features": {"populations": ["pregnancy"]}}
    with pytest.raises(ValueError, match="populations=pregnancy"):
        assessment.validate_explicit_scope(scope_answer(), query, refs)
    assert assessment.validate_explicit_scope(scope_answer(evidence_numbers=[2]), query, refs)["applied"]
    contextual = {"features": {"enzymes": ["CYP3A4"], "transporters": ["P-gp"], "process_types": ["renal_excretion"]}}
    assert assessment.validate_explicit_scope(scope_answer(), contextual, refs)["checked_constraints"] == []


@pytest.mark.parametrize("answer", [scope_answer(3, None), scope_answer(2, None), scope_answer(1, True)])
def test_scope_guard_applies_to_supported_use_grades_or_positive_applicability(answer):
    with pytest.raises(ValueError, match="renal_impairment"):
        assessment.validate_explicit_scope(answer, {"features": {"populations": ["renal_impairment"]}},
                                           [{"quote": "The model was evaluated in healthy adults with normal renal function."}])


def test_scope_guard_leaves_reference_unknown_and_negative_judgments_unchanged():
    query = {"features": {"populations": ["pregnancy"], "routes": ["oral"], "species": ["human"]}}
    for answer in (scope_answer(1, False), scope_answer(1, None), scope_answer(None, None, []), scope_answer(0, False)):
        before = copy.deepcopy(answer)
        result = assessment.validate_explicit_scope(answer, query, [])
        assert result["applied"] is False and answer == before


@pytest.mark.parametrize("dimension,value,prose", [
    ("populations", "pregnancy", "The model evaluated exposure during pregnancy."),
    ("populations", "pediatric", "The model evaluated exposure in children."),
    ("populations", "renal_impairment", "The model evaluated exposure in patients with renal impairment."),
    ("routes", "intravenous", "The model evaluated exposure after intravenous doses."),
    ("species", "human", "The model evaluated exposure in healthy volunteers."),
])
def test_partial_support_needs_cited_requested_scope_without_rule_assigned_labels(dimension, value, prose):
    answer = scope_answer(2, None)
    before = copy.deepcopy(answer)
    query = {"features": {dimension: [value]}}
    refs = [{"quote": "This repository contains a midazolam PBPK model."}, {"quote": prose}]
    with pytest.raises(ValueError, match=f"{dimension}={value}"):
        assessment.validate_explicit_scope(answer, query, refs)
    assert answer == before
    answer["evidence_numbers"] = [2]
    before = copy.deepcopy(answer)
    assert assessment.validate_explicit_scope(answer, query, refs)["applied"] is True
    assert answer == before  # The guard never upgrades U to Y or changes grade 2.


def test_scope_guard_does_not_confuse_preclinical_or_substrings_with_human_route_support():
    refs = [{"quote": "The preclinical model used rats and an oral route."}]
    with pytest.raises(ValueError, match="species=human"):
        assessment.validate_explicit_scope(scope_answer(), {"features": {"species": ["human"]}}, refs)
    with pytest.raises(ValueError, match="routes=intravenous"):
        assessment.validate_explicit_scope(scope_answer(), {"features": {"routes": ["intravenous"]}}, refs)


def test_scope_guard_alias_hash_is_reproducible_and_named():
    expected = assessment.sha(json.dumps(assessment.SCOPE_ALIASES, sort_keys=True, separators=(",", ":")).encode())
    assert assessment.SCOPE_ALIASES_SHA256 == expected
    assert "necessary-condition" in assessment.SCOPE_GUARD_VERSION


def ddi_scope_refs():
    """The actual short README that exposed the v4 same-drug scope mistakes."""
    return [
        {"quote": "# Clarithromycin-Midazolam-DDI\n"
                  "Modeling of published clinical Clarithromycin-Midazolam-DDI studies for model evaluation\n"},
        {"quote": "- [Clarithromycin OSP PBPK model](https://github.com/Open-Systems-Pharmacology/Clarithromycin-Model)\n"
                  "- [Midazolam OSP PBPK model](https://github.com/Open-Systems-Pharmacology/Midazolam-Model)\n"},
    ]


@pytest.mark.parametrize("grade,applicable", [(2, None), (2, True), (3, None), (1, True)])
@pytest.mark.parametrize("task,features", [
    ("plasma_concentration_time_prediction", {"endpoints": ["cmax"]}),
    ("special_population_prediction", {"species": ["human"], "populations": []}),
])
def test_ddi_title_and_same_drug_links_do_not_support_a_different_main_use(grade, applicable, task, features):
    answer = scope_answer(grade, applicable, [1, 2])
    before = copy.deepcopy(answer)
    query = {"compounds": ["midazolam"], "task_type": task, "features": features}
    with pytest.raises(ValueError, match=f"main_task={task}"):
        assessment.validate_explicit_scope(answer, query, ddi_scope_refs(),
                                           model_compounds=["clarithromycin", "midazolam", "rifampicin"])
    assert answer == before  # Reject unsupported claims; never assign a replacement grade.


def test_unspecified_special_population_needs_more_than_healthy_adult_pk():
    query = {"task_type": "special_population_prediction", "features": {"populations": [], "species": ["human"]}}
    refs = [{"quote": "The model describes concentration profiles in healthy adult volunteers."},
            {"quote": "The model was evaluated in patients with renal impairment."}]
    with pytest.raises(ValueError, match="main_task=special_population_prediction"):
        assessment.validate_explicit_scope(scope_answer(2, None), query, refs)
    answer = scope_answer(2, None, [2])
    before = copy.deepcopy(answer)
    result = assessment.validate_explicit_scope(answer, query, refs)
    assert result["applied"] and not result["semantic_entailment_verified"]
    assert answer == before


@pytest.mark.parametrize("endpoint,prose", [
    ("cmax", "The model predicts plasma concentration profiles and Cmax."),
    ("cmax", "The model predicts concentration profiles and C_max."),
    ("cmax", "The model predicts concentration profiles and C<sub>max</sub>."),
    ("cmax", "The model predicts peak plasma concentration."),
    ("cmax", "The model predicts maximum serum concentration."),
    ("auc", "The model predicts concentration profiles and AUC0-24."),
    ("auc", "The model predicts the area under the concentration-time curve."),
    ("auc", "The model predicts the area under the plasma concentration–time curve."),
    ("concentration_time", "The model predicts concentration–time profiles."),
    ("concentration time", "The model predicts concentration versus time."),
])
def test_actual_endpoint_prose_and_common_synonyms_pass_the_necessary_guard(endpoint, prose):
    query = {"task_type": "plasma_concentration_time_prediction", "features": {"endpoints": [endpoint]}}
    answer = scope_answer(2, None)
    before = copy.deepcopy(answer)
    result = assessment.validate_explicit_scope(answer, query, [{"quote": prose}])
    assert {"dimension": "main_task", "value": query["task_type"]} in result["checked_constraints"]
    assert {"dimension": "endpoints", "value": endpoint} in result["checked_constraints"]
    assert result["semantic_entailment_verified"] is False and answer == before


def test_endpoint_and_main_use_must_come_from_selected_prose_not_other_sources():
    query = {"task_type": "plasma_concentration_time_prediction", "features": {"endpoints": ["auc", "cmax"]}}
    refs = [*ddi_scope_refs(),
            {"quote": "The model describes plasma concentration profiles and AUC."},
            {"quote": "The model also predicts peak plasma concentration."}]
    with pytest.raises(ValueError, match="main_task=plasma_concentration_time_prediction"):
        assessment.validate_explicit_scope(scope_answer(2, True, [1, 2]), query, refs)
    with pytest.raises(ValueError, match="endpoints=cmax"):
        assessment.validate_explicit_scope(scope_answer(2, True, [3]), query, refs)
    result = assessment.validate_explicit_scope(scope_answer(2, True, [3, 4]), query, refs)
    assert result["applied"]


def test_endpoint_heading_and_url_are_not_endpoint_support():
    query = {"task_type": "plasma_concentration_time_prediction", "features": {"endpoints": ["cmax"]}}
    refs = [{"quote": "# Cmax results\nThe model describes concentration profiles.\n"
                     "[Maximum plasma concentration](https://example.test/Cmax)"}]
    with pytest.raises(ValueError, match="endpoints=cmax"):
        assessment.validate_explicit_scope(scope_answer(2, True), query, refs)


@pytest.mark.parametrize("answer", [scope_answer(0, None, [1]), scope_answer(None, False, [1]), scope_answer(1, False, [1])])
def test_registered_rifampicin_cannot_be_excluded_by_a_title_that_omits_it(answer):
    query = {"compounds": ["rifampicin"], "task_type": "plasma_concentration_time_prediction",
             "features": {"endpoints": ["auc"]}}
    before = copy.deepcopy(answer)
    with pytest.raises(ValueError, match="Unsupported compound exclusion.*rifampicin"):
        assessment.validate_explicit_scope(answer, query, ddi_scope_refs(),
                                           model_compounds=["Clarithromycin", "Midazolam", "Rifampicin"])
    assert answer == before


def test_registered_compound_exclusion_can_use_actual_target_specific_prose():
    query = {"compounds": ["rifampicin"], "task_type": "ddi_exposure_prediction", "features": {"species": ["rat"]}}
    refs = [*ddi_scope_refs(),
            {"quote": "The rifampicin simulations describe humans; this model is not applicable to rats."}]
    with pytest.raises(ValueError, match="Unsupported compound exclusion"):
        assessment.validate_explicit_scope(scope_answer(0, False, [1, 2]), query, refs,
                                           model_compounds=["rifampicin"])
    answer = scope_answer(0, False, [3])
    before = copy.deepcopy(answer)
    result = assessment.validate_explicit_scope(answer, query, refs, model_compounds=["rifampicin"])
    assert result["applied"] is False and answer == before


def test_another_drug_direct_prediction_negative_is_not_blocked_by_registered_compound_guard():
    query = {"compounds": ["metoprolol"], "task_type": "plasma_concentration_time_prediction", "features": {}}
    answer = scope_answer(0, False, [1])
    before = copy.deepcopy(answer)
    result = assessment.validate_explicit_scope(answer, query, ddi_scope_refs(),
                                               model_compounds=["clarithromycin", "midazolam", "rifampicin"])
    assert result["applied"] is False and answer == before


def test_compound_name_substrings_cannot_supply_missing_target_specific_evidence():
    query = {"compounds": ["rifampicin"], "features": {}}
    refs = [{"quote": "The synthetic identifier trifampicinX is unrelated metadata."}]
    with pytest.raises(ValueError, match="Unsupported compound exclusion"):
        assessment.validate_explicit_scope(scope_answer(0, False), query, refs, model_compounds=["rifampicin"])


def supplement_fixture(tmp_path, monkeypatch):
    _, _, out, target = scoped_fixture(tmp_path, monkeypatch)
    queries = [{"query_id": f"q{i}", "text": "Predict exposure"} for i in range(772)]
    candidates = [{"candidate_id": f"c{i}"} for i in range(280)]
    sources = []
    for candidate in candidates:
        row = source(candidate["candidate_id"])
        row["family_id"] = "frozen-family"
        row["source_artifact"]["commit"] = "same-frozen-commit"
        if row["candidate_id"] != "c1":
            row["documents"] = []
        sources.append(row)
    supplement = copy.deepcopy(sources[0])
    supplement["documents"] = source("c0", "The same-commit model describes oral exposure in adult volunteers.")["documents"]
    jsonl(out / "inputs/queries.jsonl", queries)
    jsonl(out / "inputs/candidates.jsonl", candidates)
    jsonl(target / "sources.jsonl", sources)
    return target, sources, supplement


def test_supplement_fills_only_missing_material_and_freezes_without_mutating_originals(tmp_path, monkeypatch):
    target, originals, supplement = supplement_fixture(tmp_path, monkeypatch)
    base_path = target / "sources.jsonl"
    before = base_path.read_bytes()
    supplement_path = target / "source_supplements/sources.jsonl"
    jsonl(supplement_path, [supplement])
    first = assessment.prepare()
    frozen = (target / "judge_protocol.json").read_bytes()
    assert len(first[0]) * len(first[1]) == 216160
    assert first[2]["c0"]["refs"][0]["quote"] == supplement["documents"][0]["segments"][0]["excerpt"]
    assert first[2]["c1"] == assessment.build_packet(originals[1])
    assert first[3]["input_sha256"]["assessment/source_supplements/sources.jsonl"] == assessment.file_sha(supplement_path)
    assert assessment.prepare() == first
    assert (target / "judge_protocol.json").read_bytes() == frozen
    assert base_path.read_bytes() == before
    # Even a whitespace-only change to the external packet file is a new frozen
    # input, rather than quietly changing material after judgments have begun.
    with supplement_path.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Frozen judge protocol changed"):
        assessment.prepare()
    assert (target / "judge_protocol.json").read_bytes() == frozen
    assert base_path.read_bytes() == before


@pytest.mark.parametrize("fault", ["foreign", "duplicate", "overwrite", "empty", "family_id", "family_name", "compounds", "artifact", "unverified"])
def test_supplement_cannot_replace_source_or_change_frozen_identity(tmp_path, monkeypatch, fault):
    target, originals, supplement = supplement_fixture(tmp_path, monkeypatch)
    rows = [supplement]
    if fault == "foreign":
        supplement["candidate_id"] = "outside-candidate-pool"
    elif fault == "duplicate":
        rows.append(copy.deepcopy(supplement))
    elif fault == "overwrite":
        supplement["candidate_id"] = "c1"
    elif fault == "empty":
        supplement["documents"] = []
    elif fault in {"family_id", "family_name"}:
        supplement[fault] = "different-model-family"
    elif fault == "compounds":
        supplement["compounds"] = ["another-drug"]
    elif fault == "artifact":
        supplement["source_artifact"]["commit"] = "different-commit"
    else:
        supplement["source_artifact"]["verified"] = False
    original_bytes = (target / "sources.jsonl").read_bytes()
    jsonl(target / "source_supplements/sources.jsonl", rows)
    with pytest.raises(ValueError):
        assessment.prepare()
    assert not (target / "judge_protocol.json").exists()
    assert (target / "sources.jsonl").read_bytes() == original_bytes


@pytest.mark.parametrize("initial_rank_state", ["scoring", "complete"])
def test_pipeline_loads_no_judge_until_ranking_queue_has_exited(tmp_path, monkeypatch, initial_rank_state):
    args, run, events, children = pipeline_fixture(tmp_path, monkeypatch)
    rank_status = assessment.OUT / "results/ranker/status.json"
    rank_status.write_text(json.dumps({"state": initial_rank_state}))
    rank_run = run / "rankings"
    rank_run.mkdir()
    (rank_run / "pid.json").write_text(json.dumps({"pid": 42, "start_ticks": "fixture"}))
    live = {"rank": True}
    monkeypatch.setattr(assessment, "alive", lambda saved: saved.get("pid") == 42 and live["rank"])
    def finish_rank(seconds):
        assert not children and not events
        state = json.loads((run / "pipeline_status.json").read_text())
        assert state["stage"] == "waiting_for_rankings_before_assessment"
        assert state["gpu_state"] == "not_loaded" and state["worker_pid"] is None
        rank_status.write_text('{"state":"complete"}')
        live["rank"] = False
        events.append("rank_completed_and_exited")
    monkeypatch.setattr(assessment.time, "sleep", finish_rank)
    assessment.pipeline(args)
    assert events[:2] == ["rank_completed_and_exited", "primary"]
    assert json.loads((run / "pipeline_status.json").read_text())["state"] == "complete"


def test_pipeline_does_not_bypass_an_interrupted_ranking_queue(tmp_path, monkeypatch):
    args, run, events, children = pipeline_fixture(tmp_path, monkeypatch)
    (assessment.OUT / "results/ranker/status.json").write_text('{"state":"scoring"}')
    assessment.pipeline(args)
    state = json.loads((run / "pipeline_status.json").read_text())
    assert state["state"] == "blocked_rankings_not_running"
    assert state["incomplete_ranking_methods"] == ["ranker"]
    assert not children and not events


def test_pipeline_user_stop_during_cpu_queue_does_not_start_a_judge(tmp_path, monkeypatch):
    args, run, events, children = pipeline_fixture(tmp_path, monkeypatch)
    (assessment.OUT / "results/ranker/status.json").write_text('{"state":"scoring"}')
    rank_run = run / "rankings"
    rank_run.mkdir()
    (rank_run / "pid.json").write_text(json.dumps({"pid": 42, "start_ticks": "fixture"}))
    monkeypatch.setattr(assessment, "alive", lambda saved: saved.get("pid") == 42)
    monkeypatch.setattr(assessment.time, "sleep", lambda seconds: (run / "stop.requested").touch())
    assessment.pipeline(args)
    assert json.loads((run / "pipeline_status.json").read_text())["state"] == "stopped"
    assert not children and not events
