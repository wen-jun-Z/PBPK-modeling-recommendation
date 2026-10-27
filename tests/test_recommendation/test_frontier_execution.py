"""CPU regressions for complete-pool, resumable batched ranking execution."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_frontier_comparison as runner
from frontier_external_models import GenerativeTop20Reranker, RankR1Top20Reranker


def queries(count=7):
    return [{"query_id": f"q{i}", "text": f"original:{i}", "variants": {
        variant: {"text": f"{variant}:{i}"} for variant in runner.VARIANTS[1:]
    }} for i in range(count)]


class BatchAdapter:
    def __init__(self):
        self.calls = []
        self.last_query_diagnostics = {}

    def score_batch(self, texts, documents):
        self.calls.append(list(texts))
        values = []
        diagnostics = []
        for text in texts:
            variant, index = text.split(":")
            offset = runner.VARIANTS.index(variant) * 10 + int(index)
            values.append(np.arange(len(documents), dtype=np.float32) + offset)
            diagnostics.append({"generation_valid": True, "response": text})
        return {"scores": np.asarray(values), "diagnostics": diagnostics,
                "batch_seconds": 999.0}

    def score_query(self, text, documents):
        result = self.score_batch([text], documents)
        self.last_query_diagnostics = result["diagnostics"][0]
        return result["scores"][0]


def test_scored_batches_oom_splits_without_duplicate_yields(monkeypatch):
    class AllocationBoundAdapter(BatchAdapter):
        def score_batch(self, texts, documents):
            if len(texts) > 2:
                self.calls.append(list(texts))
                raise torch.OutOfMemoryError("fake allocation failure")
            return super().score_batch(texts, documents)

    emptied = []
    guarded = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    adapter = AllocationBoundAdapter()
    pending = [0, 2, 3, 5, 6]
    batches = list(runner.scored_batches(
        adapter, queries(), ["a", "b", "c"], pending, "reordered", 4,
        lambda: guarded.append(True)))
    assert [i for batch in batches for i in batch[0]] == pending
    assert [len(batch[0]) for batch in batches] == [2, 2, 1]
    assert all(batch[4] == 1 for batch in batches)
    assert len(emptied) == 1 and len(guarded) == 4
    assert adapter.calls == [
        ["reordered:0", "reordered:2", "reordered:3", "reordered:5"],
        ["reordered:0", "reordered:2"], ["reordered:3", "reordered:5"], ["reordered:6"],
    ]
    assert all(0 < batch[3] < 999 for batch in batches)


@pytest.mark.parametrize("fault", ["shape", "nonfinite", "diagnostics"])
def test_scored_batches_rejects_incomplete_outputs_before_yield(fault):
    class InvalidAdapter(BatchAdapter):
        def score_batch(self, texts, documents):
            result = super().score_batch(texts, documents)
            if fault == "shape":
                result["scores"] = result["scores"][:, :-1]
            elif fault == "nonfinite":
                result["scores"][0, 0] = np.nan
            else:
                result["diagnostics"] = result["diagnostics"][:-1]
            return result

    with pytest.raises(RuntimeError):
        next(runner.scored_batches(
            InvalidAdapter(), queries(), ["a", "b"], [0, 2], "original", 2, lambda: None))


def test_single_query_oom_is_not_retried(monkeypatch):
    class AlwaysOOM(BatchAdapter):
        def score_batch(self, texts, documents):
            self.calls.append(list(texts))
            raise torch.OutOfMemoryError("fake allocation failure")

    adapter = AlwaysOOM()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("no retry at size one"))
    with pytest.raises(torch.OutOfMemoryError):
        next(runner.scored_batches(adapter, queries(), ["a"], [3], "original", 2, lambda: None))
    assert adapter.calls == [["original:3"]]


def test_run_method_resumes_each_variant_and_records_amortized_timing(monkeypatch, tmp_path):
    out, run = tmp_path / "results_root", tmp_path / "runtime"
    run.mkdir()
    dest = out / "results" / "bm25_text"
    dest.mkdir(parents=True)
    q = queries(4)
    candidates = [{"candidate_id": f"c{i}", "text": f"document {i}"} for i in range(3)]
    old_scores = np.full((3, 4, 3), np.nan, dtype=np.float32)
    old_times = np.full((3, 4), np.nan, dtype=np.float64)
    completed = [(0, 0), (0, 2), (1, 1), (1, 2), (1, 3)]
    for vi, qi in completed:
        old_scores[vi, qi] = -777
        old_times[vi, qi] = 7
    old_scores[1, 0] = 999  # Scores without a finite timing commit must rerun.
    np.save(dest / "scores.npy", old_scores)
    np.save(dest / "query_seconds.npy", old_times)
    (dest / "batch_benchmark.json").write_text(json.dumps({"selected_query_batch_size": 2}))
    (dest / "generation_diagnostics.jsonl").write_text("".join(
        json.dumps({"query_id": q[qi]["query_id"], "variant": runner.VARIANTS[vi],
                    "generation_valid": True, "response": "preserved completed fixture"}) + "\n"
        for vi, qi in completed))
    adapter = BatchAdapter()
    ticks = iter(range(100))
    monkeypatch.setattr(runner, "OUT", out)
    monkeypatch.setattr(runner, "RUN", run)
    monkeypatch.setattr(runner, "BM25", lambda documents: adapter)
    monkeypatch.setattr(runner, "load_inputs", lambda: (q, candidates, {"frozen": "unchanged"}))
    monkeypatch.setattr(runner, "report", lambda: None)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**40))
    monkeypatch.setattr(runner, "time", SimpleNamespace(
        perf_counter=lambda: float(next(ticks)), monotonic=time.monotonic))

    runner.run_method("bm25_text", query_batch_size=4, benchmark_batches=True)
    assert adapter.calls == [["original:1", "original:3"], ["reordered:0"],
                             ["missing_evidence:0", "missing_evidence:1"],
                             ["missing_evidence:2", "missing_evidence:3"]]
    scores = np.load(dest / "scores.npy")
    timings = np.load(dest / "query_seconds.npy")
    assert np.isfinite(scores).all() and np.isfinite(timings).all()
    for vi, qi in completed:
        assert np.array_equal(scores[vi, qi], [-777] * 3)
        assert timings[vi, qi] == 7
    assert np.array_equal(scores[1, 0], [10, 11, 12])
    assert timings[0, 1] == timings[0, 3] == .5
    assert timings[1, 0] == 1
    batches = [json.loads(line) for line in (dest / "execution_batches.jsonl").read_text().splitlines()]
    assert sum(len(row["query_ids"]) for row in batches) == 7
    assert batches[0]["query_seconds_semantics"] == "amortized_batch_wall_seconds"
    assert batches[1]["query_seconds_semantics"] == "single_query_wall_seconds"
    status = json.loads((dest / "status.json").read_text())
    assert status["state"] == "complete" and status["completed_query_variants"] == 12
    assert status["query_batch_size_effective_initial"] == 2
    prior_calls = list(adapter.calls)
    runner.run_method("bm25_text", query_batch_size=2)
    assert adapter.calls == prior_calls


def test_batch_benchmark_records_fixed_input_comparison(tmp_path):
    adapter = BatchAdapter()
    selected = runner.benchmark_query_batches(adapter, queries(), ["a", "b"], 4, tmp_path)
    record = json.loads((tmp_path / "batch_benchmark.json").read_text())
    assert selected == 4
    assert record["query_ids"] == ["q0", "q2", "q4", "q6"]
    assert record["warmup_excluded"] == 1
    assert [row["query_batch_size"] for row in record["batch_attempts"]] == [2, 4]
    assert all(row["identical_score_rows"] == 4 for row in record["batch_attempts"])
    assert all(row["identical_response_rows"] == 4 for row in record["batch_attempts"])


def test_batch_benchmark_rejects_missing_diagnostics(tmp_path):
    class MissingDiagnostics(BatchAdapter):
        def score_batch(self, texts, documents):
            result = super().score_batch(texts, documents)
            if len(texts) > 1:
                result["diagnostics"] = result["diagnostics"][:-1]
            return result

    with pytest.raises(RuntimeError, match="coverage mismatch"):
        runner.benchmark_query_batches(MissingDiagnostics(), queries(), ["a", "b"], 2, tmp_path)


def test_setwise_benchmark_compares_winner_responses_not_absent_top_level_field(tmp_path):
    class SetwiseDiagnostics(BatchAdapter):
        def score_batch(self, texts, documents):
            result = super().score_batch(texts, documents)
            result["diagnostics"] = [{"generation_valid": True, "winner_calls": [
                {"response": f"batch size {len(texts)}"}]} for _ in texts]
            return result

    runner.benchmark_query_batches(SetwiseDiagnostics(), queries(), ["a", "b"], 2, tmp_path)
    record = json.loads((tmp_path / "batch_benchmark.json").read_text())
    assert record["batch_attempts"][0]["identical_response_rows"] == 0


class CapacityAdapter:
    """Deterministic first-stage timings and device capacity, without loading CUDA."""
    SYSTEM_PROMPT = "system"
    USER_PROMPT = "{query}\n{docs}"
    max_input_tokens = 10000
    max_new_tokens = 2048

    def __init__(self, seconds=None, free_gib=None, oom_at=None, missing=None):
        self.calls, self.elapsed, self.current_size = [], 0.0, 1
        self.seconds = seconds or {1: 10, 4: 4, 8: 2, 16: 3, 32: 1}
        self.free_gib = free_gib or {1: 20, 4: 20, 8: 12, 16: 8, 32: 3}
        self.oom_at, self.missing = oom_at, missing
        self.bm25 = SimpleNamespace(score_query=lambda text, docs: np.arange(len(docs), 0, -1))
        self.tokenizer = SimpleNamespace()  # Special-method lookup needs a real class.
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(row["content"] for row in messages)

            def __call__(self, prompts, **kwargs):
                return {"input_ids": [list(range(len(prompt))) for prompt in prompts]}
        self.tokenizer = Tokenizer()
        self.torch = SimpleNamespace(OutOfMemoryError=torch.OutOfMemoryError, cuda=SimpleNamespace(
            empty_cache=lambda: None, reset_peak_memory_stats=lambda: None,
            mem_get_info=lambda: (int(self.free_gib[self.current_size] * 2**30), 48 * 2**30),
            max_memory_allocated=lambda: 5 * 2**30, max_memory_reserved=lambda: 8 * 2**30))

    def prepare_documents(self, documents):
        self.documents = list(documents)

    def _passage(self, text):
        return text

    def _sync(self):
        pass

    def _generate_prompts(self, prompts):
        self.calls.append(list(prompts))
        self.current_size = len(prompts)
        if self.oom_at is not None and len(prompts) >= self.oom_at:
            raise torch.OutOfMemoryError("fake capacity limit")
        self.elapsed += len(prompts) * self.seconds[len(prompts)]
        output = [{"response": "invalid" if len(prompts) == 8 else "<answer>[1]</answer>",
                   "generated_token_count": 10, "hit_token_limit": False} for _ in prompts]
        if len(prompts) > 1 and self.missing == "short":
            output.pop()
        if len(prompts) > 1 and self.missing == "none":
            output[0] = None
        return output

    def parse_winner(self, text, count):
        return 0 if text.startswith("<answer>") else None


def capacity_setup(monkeypatch, tmp_path, **kwargs):
    adapter = CapacityAdapter(**kwargs)
    monkeypatch.setattr(runner, "RUN", tmp_path)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**40))
    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=lambda: adapter.elapsed))
    rows = [{"query_id": f"q{i}", "text": "x" * (i + 1), "unused_relevance_label": 39 - i}
            for i in range(40)]
    return adapter, rows


def test_capacity_compares_same_longest_inputs_and_selects_fastest_safe_batch(monkeypatch, tmp_path):
    adapter, rows = capacity_setup(monkeypatch, tmp_path)
    selected = runner.tune_rank_r1_batches(adapter, rows, ["a", "b"], 32, tmp_path, {"input": "fixed"})
    record = json.loads((tmp_path / "batch_capacity.json").read_text())
    assert selected == 8  # 32 is faster but lacks headroom; 16 has more valid answers but is slower.
    assert record["query_ids"] == [f"q{i}" for i in range(39, 7, -1)]
    assert record["warmup_excluded"] == 1 and len(adapter.calls[0]) == 1
    assert [row["query_batch_size"] for row in record["attempts"]] == [4, 8, 16, 32]
    by_size = {size: [prompt for call in adapter.calls[1:] if len(call) == size for prompt in call]
               for size in (4, 8, 16, 32)}
    assert len(by_size[4]) == 32
    assert by_size[4] == by_size[8] == by_size[16] == by_size[32]
    assert record["attempts"][1]["valid_winner_outputs"] == 0
    assert all(row["queries"] == 32 for row in record["attempts"])


@pytest.mark.parametrize("oom_at,selected,attempts", [(16, 8, [4, 8, 16]), (4, 1, [4])])
def test_capacity_oom_stops_growth_and_keeps_verified_fallback(monkeypatch, tmp_path, oom_at, selected, attempts):
    adapter, rows = capacity_setup(monkeypatch, tmp_path, oom_at=oom_at)
    assert runner.tune_rank_r1_batches(adapter, rows, ["d"], 32, tmp_path, {}) == selected
    record = json.loads((tmp_path / "batch_capacity.json").read_text())
    assert [row["query_batch_size"] for row in record["attempts"]] == attempts
    assert record["attempts"][-1]["state"] == "out_of_memory"
    assert max(map(len, adapter.calls)) == oom_at


def test_capacity_no_batch_with_headroom_uses_smallest_completed(monkeypatch, tmp_path):
    adapter, rows = capacity_setup(monkeypatch, tmp_path, free_gib={1: 3, 4: 3, 8: 2, 16: 1, 32: .5})
    assert runner.tune_rank_r1_batches(adapter, rows, ["d"], 32, tmp_path, {}) == 4


@pytest.mark.parametrize("missing", ["short", "none"])
def test_capacity_rejects_missing_outputs(monkeypatch, tmp_path, missing):
    adapter, rows = capacity_setup(monkeypatch, tmp_path, missing=missing)
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        runner.tune_rank_r1_batches(adapter, rows, ["d"], 32, tmp_path, {})
    assert json.loads((tmp_path / "batch_capacity.json").read_text())["state"] != "complete"


def test_capacity_identical_settings_reuse_cache_without_generation(monkeypatch, tmp_path):
    adapter, rows = capacity_setup(monkeypatch, tmp_path)
    assert runner.tune_rank_r1_batches(adapter, rows, ["d"], 32, tmp_path, {"input": "fixed"}) == 8
    prior_calls = list(adapter.calls)
    assert runner.tune_rank_r1_batches(adapter, rows, ["d"], 32, tmp_path, {"input": "fixed"}) == 8
    assert adapter.calls == prior_calls
    runner.tune_rank_r1_batches(adapter, rows, ["d"], 16, tmp_path, {"input": "fixed"})
    assert len(adapter.calls) > len(prior_calls)


def test_run_method_capacity_tuning_overrides_old_benchmark_without_overwriting_commits(monkeypatch, tmp_path):
    out, run = tmp_path / "results_root", tmp_path / "runtime"
    run.mkdir()
    dest = out / "results" / "bm25_text"
    scores, timings = journal_arrays(dest, count=10)
    scores[0, 0], timings[0, 0] = [222, 111], 13
    scores.flush(); timings.flush()
    (dest / "generation_diagnostics.jsonl").write_text(json.dumps(journal_record("q0")) + "\n")
    (dest / "batch_benchmark.json").write_text(json.dumps({"selected_query_batch_size": 4}))
    q = queries(10)
    adapter = BatchAdapter()
    adapter.total_winner_calls = 7
    tuned = []

    def tune(active, rows, documents, requested, folder, hashes):
        assert active is adapter and rows == q and requested == 32
        assert folder == dest and hashes == {"input": "fixed"}
        adapter.total_winner_calls += 32
        tuned.append(True)
        return 8

    monkeypatch.setattr(runner, "OUT", out)
    monkeypatch.setattr(runner, "RUN", run)
    monkeypatch.setattr(runner, "METHODS", {"bm25_text": {"kind": "rank_r1"}})
    monkeypatch.setattr(runner, "BM25", lambda documents: adapter)
    monkeypatch.setattr(runner, "load_inputs", lambda: (
        q, [{"candidate_id": "a", "text": "a"}, {"candidate_id": "b", "text": "b"}], {"input": "fixed"}))
    monkeypatch.setattr(runner, "tune_rank_r1_batches", tune)
    monkeypatch.setattr(runner, "benchmark_query_batches", lambda *a: pytest.fail("old benchmark must not run"))
    monkeypatch.setattr(runner, "report", lambda: None)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**40))
    runner.run_method("bm25_text", query_batch_size=32, benchmark_batches=True, tune_query_batches=True)
    assert tuned == [True] and adapter.total_winner_calls == 7
    assert [len(call) for call in adapter.calls] == [8, 1, 8, 2, 8, 2]
    assert np.array_equal(np.load(dest / "scores.npy")[0, 0], [222, 111])
    assert np.load(dest / "query_seconds.npy")[0, 0] == 13
    status = json.loads((dest / "status.json").read_text())
    assert status["state"] == "complete" and status["query_batch_size_effective_initial"] == 8


class TensorInputs(dict):
    def to(self, device):
        return TensorInputs({key: value.to(device) for key, value in self.items()})


class FakeTokenizer:
    padding_side = "left"
    pad_token_id = 9
    eos_token_id = 2

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = [[ord(char) + 10 for char in text] for text in texts]
        width = max(map(len, rows))
        return TensorInputs(
            input_ids=torch.tensor([[9] * (width - len(row)) + row for row in rows]),
            attention_mask=torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows]))

    def decode(self, tokens, **kwargs):
        return "".join(chr(int(token) - 10) for token in tokens if int(token) not in (2, 3, 9))

    def tokenize(self, text):
        return list(text)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)


class FakeGenerator:
    generation_config = SimpleNamespace(eos_token_id=[2, 3])

    def __init__(self, tokenizer, callback):
        self.tokenizer, self.callback = tokenizer, callback
        self.batch_sizes = []

    def generate(self, input_ids, attention_mask, **kwargs):
        assert kwargs["do_sample"] is False and kwargs["use_cache"]
        assert kwargs["logits_to_keep"] == 1
        self.batch_sizes.append(len(input_ids))
        responses = []
        for index, row in enumerate(input_ids):
            prompt = self.tokenizer.decode(row[attention_mask[index].bool()])
            response = self.callback(prompt)
            responses.append([ord(char) + 10 for char in response] + [2 if index % 2 == 0 else 3])
        width = max(map(len, responses))
        tails = torch.tensor([row + [9] * (width - len(row)) for row in responses])
        return torch.cat([input_ids, tails], dim=1)


def generative_adapter(cls, callback, documents):
    adapter = cls.__new__(cls)
    adapter.tokenizer = FakeTokenizer()
    adapter.model = FakeGenerator(adapter.tokenizer, callback)
    adapter.torch, adapter.device = torch, "cpu"
    adapter.bm25 = SimpleNamespace(score_query=lambda query, docs: np.arange(len(docs), 0, -1, dtype=np.float32))
    adapter._documents = tuple(documents)
    adapter.max_input_tokens, adapter.max_new_tokens = 14336, 2048
    adapter.valid_generation_count = adapter.invalid_generation_count = adapter.total_winner_calls = 0
    adapter.tokenized_input_count = adapter.truncated_input_count = 0
    adapter.passage_token_limit, adapter.per_query_budget_seconds = 512, 1800
    adapter.reasonrank = False
    return adapter


def test_generation_slices_after_padding_and_counts_through_first_eos():
    adapter = generative_adapter(GenerativeTop20Reranker, lambda text: "abc" if text == "short" else "abcdefg", ["d"])
    outputs = adapter._generate_prompts(["short", "muchlonger"])
    assert [row["response"] for row in outputs] == ["abc", "abcdefg"]
    assert [row["input_tokens"] for row in outputs] == [5, 10]
    assert [row["generated_token_count"] for row in outputs] == [4, 8]
    adapter.max_input_tokens = 7
    with pytest.raises(ValueError, match="frozen input budget"):
        adapter._generate_prompts(["short", "muchlonger"])
    outputs = adapter._generate_prompts(["short", "muchlonger"], allow_over_budget=True)
    assert outputs[0]["response"] == "abc" and outputs[1] is None


def test_listwise_batch_preserves_full_pool_and_strict_fallback():
    response = "<answer>" + " > ".join(f"[{i}]" for i in range(20, 0, -1)) + "</answer>"
    adapter = generative_adapter(
        GenerativeTop20Reranker, lambda prompt: "<answer>[1]</answer>" if "badquery" in prompt else response,
        [f"doc{i}" for i in range(25)])
    result = adapter.score_batch(["goodquery", "badquery"])
    assert [row["generation_valid"] for row in result["diagnostics"]] == [True, False]
    assert np.argsort(-result["scores"][0]).tolist() == list(range(19, -1, -1)) + list(range(20, 25))
    assert np.argsort(-result["scores"][1]).tolist() == list(range(25))
    assert np.array_equal(adapter.score_query("goodquery"), result["scores"][0])
    assert (adapter.valid_generation_count, adapter.invalid_generation_count) == (2, 1)


def test_rank_r1_removes_failed_request_without_corrupting_other_order():
    counts = {}

    def winner(prompt):
        key = "badquery" if "badquery" in prompt else "goodquery"
        counts[key] = counts.get(key, 0) + 1
        choice = "[99]" if key == "badquery" and counts[key] == 2 else "[2]"
        return f"<think>x</think><answer>{choice}</answer>"

    adapter = generative_adapter(RankR1Top20Reranker, winner, [f"d{i}" for i in range(4)])
    result = adapter.score_batch(["goodquery", "badquery"])
    assert [row["generation_valid"] for row in result["diagnostics"]] == [True, False]
    assert [row["winner_call_count"] for row in result["diagnostics"]] == [3, 2]
    assert result["diagnostics"][0]["selected_candidate_indices"] == [1, 2, 3, 0]
    assert np.argsort(-result["scores"][1]).tolist() == [0, 1, 2, 3]
    assert adapter.model.batch_sizes == [2, 2, 1]
    assert adapter.total_winner_calls == 5


def test_rank_r1_mid_batch_oom_does_not_commit_generation_counters():
    calls = []

    def winner(prompt):
        calls.append(prompt)
        if len(calls) == 3:
            raise torch.OutOfMemoryError("fake second-stage OOM")
        return "<think>x</think><answer>[1]</answer>"

    adapter = generative_adapter(RankR1Top20Reranker, winner, ["a", "b", "c"])
    with pytest.raises(torch.OutOfMemoryError):
        adapter.score_batch(["q0", "q1"])
    assert (adapter.valid_generation_count, adapter.invalid_generation_count, adapter.total_winner_calls) == (0, 0, 0)


def journal_arrays(dest, count=3):
    dest.mkdir(parents=True, exist_ok=True)
    scores = np.lib.format.open_memmap(dest / "scores.npy", mode="w+", dtype="float32", shape=(3, count, 2))
    timings = np.lib.format.open_memmap(dest / "query_seconds.npy", mode="w+", dtype="float64", shape=(3, count))
    scores[:] = np.nan
    timings[:] = np.nan
    scores.flush()
    timings.flush()
    return scores, timings


def journal_record(query_id, variant="original"):
    return {"query_id": query_id, "variant": variant, "generation_valid": True,
            "response": "<answer>[1] > [2]</answer>"}


def test_journal_keeps_committed_rows_and_repairs_missing_final_newline(tmp_path):
    scores, timings = journal_arrays(tmp_path)
    scores[0, 0], timings[0, 0] = [2, 1], 7
    scores.flush()
    timings.flush()
    path = tmp_path / "generation_diagnostics.jsonl"
    row = journal_record("q0")
    path.write_text(json.dumps(row))  # Valid JSON whose final newline was interrupted.
    runner.reconcile_generation_journal(tmp_path, scores, timings, queries(3))
    assert np.array_equal(scores[0, 0], [2, 1]) and timings[0, 0] == 7
    assert path.read_text().endswith("\n")
    with path.open("a") as handle:
        handle.write(json.dumps(journal_record("q1")) + "\n")
    assert len([json.loads(line) for line in path.read_text().splitlines()]) == 2


def test_journal_archives_uncommitted_partial_and_legacy_rows_before_resume(tmp_path):
    scores, timings = journal_arrays(tmp_path)
    scores[0, 0], timings[0, 0] = [2, 1], 7
    scores[0, 1] = [91, 92]  # No timing commit: its diagnostic must be separated.
    scores[1, 2], timings[1, 2] = [21, 22], 11  # Legacy commit without a diagnostic.
    scores.flush()
    timings.flush()
    path = tmp_path / "generation_diagnostics.jsonl"
    committed, uncommitted = journal_record("q0"), journal_record("q1")
    tail = '{"query_id":"q2","response":'
    path.write_text(json.dumps(committed) + "\n" + json.dumps(uncommitted) + "\n" + tail)
    runner.reconcile_generation_journal(tmp_path, scores, timings, queries(3))
    assert [json.loads(line) for line in path.read_text().splitlines()] == [committed]
    assert np.array_equal(scores[0, 0], [2, 1]) and timings[0, 0] == 7
    assert np.array_equal(scores[1, 2], [21, 22]) and np.isnan(timings[1, 2])
    archives = [json.loads(line) for line in (tmp_path / "interrupted_generation_records.jsonl").read_text().splitlines()]
    by_reason = {row["reason"]: row for row in archives}
    assert by_reason["no_committed_score_and_timing"]["record"] == uncommitted
    assert by_reason["partial_tail"]["raw_record"] == tail
    assert by_reason["committed_array_without_generation_record"]["scores"] == [21, 22]
    assert by_reason["committed_array_without_generation_record"]["seconds"] == 11
    committed_mask = np.isfinite(scores).all(axis=2) & np.isfinite(timings)
    pending = np.flatnonzero(~committed_mask[0]).tolist()
    adapter = BatchAdapter()
    resumed = list(runner.scored_batches(adapter, queries(3), ["a", "b"], pending, "original", 2, lambda: None))
    assert [index for batch in resumed for index in batch[0]] == [1, 2]
    before = (tmp_path / "interrupted_generation_records.jsonl").read_text()
    runner.reconcile_generation_journal(tmp_path, scores, timings, queries(3))
    assert (tmp_path / "interrupted_generation_records.jsonl").read_text() == before


def test_journal_absent_file_does_not_silently_accept_legacy_array_commit(tmp_path):
    scores, timings = journal_arrays(tmp_path)
    scores[2, 1], timings[2, 1] = [31, 32], 13
    scores.flush()
    timings.flush()
    runner.reconcile_generation_journal(tmp_path, scores, timings, queries(3))
    assert np.isnan(timings[2, 1]) and np.array_equal(scores[2, 1], [31, 32])
    archived = [json.loads(line) for line in (tmp_path / "interrupted_generation_records.jsonl").read_text().splitlines()]
    assert len(archived) == 1 and archived[0]["reason"] == "committed_array_without_generation_record"
    assert archived[0]["variant"] == "missing_evidence" and archived[0]["query_id"] == "q1"
    assert archived[0]["seconds"] == 13


@pytest.mark.parametrize("fault", ["interior_corruption", "duplicate_committed"])
def test_journal_refuses_ambiguous_committed_history_without_mutating_files(tmp_path, fault):
    scores, timings = journal_arrays(tmp_path)
    scores[0, 0], timings[0, 0] = [2, 1], 7
    scores.flush()
    timings.flush()
    path = tmp_path / "generation_diagnostics.jsonl"
    complete = json.dumps(journal_record("q0")) + "\n"
    original = complete + ("{broken json\n" + complete if fault == "interior_corruption" else complete)
    path.write_text(original)
    before_scores, before_times = scores.copy(), timings.copy()
    with pytest.raises(RuntimeError, match="Corrupt generation journal interior|Duplicate committed"):
        runner.reconcile_generation_journal(tmp_path, scores, timings, queries(3))
    assert np.array_equal(scores, before_scores, equal_nan=True)
    assert np.array_equal(timings, before_times, equal_nan=True)
    assert path.read_text() == original
    assert not (tmp_path / "interrupted_generation_records.jsonl").exists()


def test_journal_archive_failure_preserves_original_timing_commit(monkeypatch, tmp_path):
    scores, timings = journal_arrays(tmp_path)
    scores[0, 0], timings[0, 0] = [2, 1], 7
    scores.flush()
    timings.flush()
    (tmp_path / "generation_diagnostics.jsonl").write_text("")
    original_open = Path.open

    def fail_archive(path, *args, **kwargs):
        if path.name == "interrupted_generation_records.jsonl":
            raise OSError("fake archive write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_archive)
    with pytest.raises(OSError, match="fake archive write failure"):
        runner.reconcile_generation_journal(tmp_path, scores, timings, queries(3))
    assert timings[0, 0] == 7 and np.array_equal(scores[0, 0], [2, 1])


def test_report_separates_batch_throughput_and_filters_execution_modes_in_groups(monkeypatch, tmp_path):
    import frontier_result_audit

    method = "bm25_top20_rearank_7b_nf4"
    dest = tmp_path / "results" / method
    dest.mkdir(parents=True)
    q = queries(4)
    for index, row in enumerate(q):
        row["task_type"] = "serial_originals" if index < 2 else "batched_originals"
        row["compound_id"] = "compound_a" if index % 2 == 0 else "compound_b"
    candidates = [{"candidate_id": f"c{i}", "text": f"document {i}"} for i in range(20)]
    descending = np.arange(20, 0, -1, dtype=np.float32)
    scores = np.tile(descending, (3, 4, 1))
    scores[1, :2] = descending[::-1]
    scores[2, :2] = descending[::-1]
    scores[2, 3] = descending[::-1]
    timings = np.full((3, 4), 2.0)
    timings[0, :2] = [30, 20]
    np.save(dest / "scores.npy", scores)
    np.save(dest / "query_seconds.npy", timings)
    (dest / "status.json").write_text(json.dumps({"state": "complete"}))
    batches = [{"variant": "original", "query_ids": ["q2", "q3"],
                "query_batch_size": 2, "batch_wall_seconds": 4.0}]
    batches.extend({"variant": variant, "query_ids": [f"q{i}", f"q{i+1}"],
                    "query_batch_size": 2, "batch_wall_seconds": 4.0}
                   for variant in runner.VARIANTS[1:] for i in (0, 2))
    (dest / "execution_batches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in batches))
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "METHODS", {method: {"kind": "rearank"}})
    monkeypatch.setattr(runner, "load_inputs", lambda: (q, candidates, {"frozen": "unchanged"}))
    monkeypatch.setattr(frontier_result_audit, "refresh_output_recovery", lambda out: {})

    entry = runner.report()["methods"][method]
    assert entry["latency"]["p50"] is None and entry["latency"]["p95"] is None
    assert entry["throughput_by_query_batch_size"] == {
        "2": {"queries": 10, "batch_wall_seconds": 20.0, "queries_per_second": .5}}
    for variant, expected in (("reordered", 1.0), ("missing_evidence", .5)):
        primary = entry["robustness"][variant]
        assert primary["changed_input_queries"] == 4
        assert primary["same_execution_batch_size_pairs"] == 2
        assert primary["cross_execution_mode_pairs_excluded"] == 2
        assert primary["top1_consistency"] == primary["top10_overlap"] == expected
        assert entry["robustness_all_recorded_execution_modes"][variant]["top1_consistency"] == expected / 2
        excluded = entry["groups"]["by_task"]["serial_originals"][variant]
        assert excluded["changed_input_queries"] == 2 and excluded["comparable_pairs"] == 0
        assert excluded["cross_execution_mode_pairs_excluded"] == 2
        assert excluded["top1_consistency"] is None and excluded["top10_overlap"] is None
        comparable = entry["groups"]["by_task"]["batched_originals"][variant]
        assert comparable["changed_input_queries"] == comparable["comparable_pairs"] == 2
        assert comparable["cross_execution_mode_pairs_excluded"] == 0
        assert comparable["top1_consistency"] == comparable["top10_overlap"] == expected
        for compound in ("compound_a", "compound_b"):
            grouped = entry["groups"]["by_compound"][compound][variant]
            assert grouped["changed_input_queries"] == 2 and grouped["comparable_pairs"] == 1
            assert grouped["cross_execution_mode_pairs_excluded"] == 1
            expected_compound = 0.0 if variant == "missing_evidence" and compound == "compound_b" else 1.0
            assert grouped["top1_consistency"] == grouped["top10_overlap"] == expected_compound
