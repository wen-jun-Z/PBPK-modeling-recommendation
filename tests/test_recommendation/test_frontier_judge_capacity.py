"""CPU-only execution checks: bounded allocator growth and intact prompt batches."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_frontier_assessment as assessment


GIB = 1024**3


def completed_rank():
    return {"state": "complete", "worker_pid": 12345, "worker_start_ticks": "789",
            "completed_query_variants": 772 * 3, "completed_pairs": 772 * 280 * 3,
            "finished_at": "fixed-time"}


def fake_proc(tmp_path, monkeypatch, *, state="S", ticks="789", different_cwd=False):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(assessment, "ROOT", root)
    proc_root = tmp_path / "proc"
    worker = proc_root / "12345"
    worker.mkdir(parents=True)
    fields = ["0"] * 22
    fields[:3] = ["12345", "(python)", state]
    fields[21] = ticks
    (worker / "stat").write_text(" ".join(fields))
    (worker / "cwd").symlink_to(tmp_path if different_cwd else root, target_is_directory=True)
    return proc_root


def test_rank_exit_requires_complete_full_run_and_exact_saved_identity(tmp_path, monkeypatch):
    proc = fake_proc(tmp_path, monkeypatch)
    assert assessment.GPUJudge._inactive_rank_worker(completed_rank(), proc) is None
    (proc / "12345/stat").write_text("12345 (python) Z " + " ".join(["0"] * 18 + ["789"]))
    proof = assessment.GPUJudge._inactive_rank_worker(completed_rank(), proc)
    assert proof["exit_confirmation"] == "same_uid_cwd_start_ticks_zombie"
    assert proof["rank_worker_start_ticks"] == "789"


@pytest.mark.parametrize("field,value", [
    ("state", "scoring"), ("completed_query_variants", 2315), ("completed_pairs", 1),
    ("worker_pid", None), ("worker_pid", True), ("worker_start_ticks", "invalid"),
])
def test_missing_pid_alone_cannot_authorize_expansion_for_bad_rank_status(tmp_path, field, value):
    status = completed_rank()
    status[field] = value
    assert assessment.GPUJudge._inactive_rank_worker(status, tmp_path / "proc") is None


def test_saved_pid_absent_confirms_exit_without_scanning_other_processes(tmp_path):
    proof = assessment.GPUJudge._inactive_rank_worker(completed_rank(), tmp_path / "proc")
    assert proof["exit_confirmation"] == "saved_pid_absent"
    assert proof["rank_worker_pid"] == 12345


@pytest.mark.parametrize("fault", ["reused_pid", "cwd", "uid", "broken_stat", "missing_cwd"])
def test_foreign_or_unverifiable_identity_fails_closed(tmp_path, monkeypatch, fault):
    proc = fake_proc(tmp_path, monkeypatch, state="Z",
                     ticks="987" if fault == "reused_pid" else "789",
                     different_cwd=fault == "cwd")
    if fault == "uid":
        actual_uid = assessment.os.getuid()
        monkeypatch.setattr(assessment.os, "getuid", lambda: actual_uid + 1)
    elif fault == "broken_stat":
        (proc / "12345/stat").write_text("truncated")
    elif fault == "missing_cwd":
        (proc / "12345/cwd").unlink()
    assert assessment.GPUJudge._inactive_rank_worker(completed_rank(), proc) is None


def cpu_judge(tmp_path, monkeypatch, *, free=18 * GIB, total=48 * GIB, reserved=10 * GIB):
    out = tmp_path / "frontier"
    status = out / "results/bm25_top20_rank_r1_3b/status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps(completed_rank()))
    monkeypatch.setattr(assessment, "OUT", out)
    judge = assessment.GPUJudge.__new__(assessment.GPUJudge)
    judge.memory_cap_gib = 12.0
    judge.prompt_batch_limit = 8
    judge.batch_token_budget = 8000
    judge.capacity_upgrade = None
    judge._prompt_length_cache = {}
    calls = []
    judge.torch = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: calls.append(("synchronize",)),
        mem_get_info=lambda _: (free, total),
        memory_reserved=lambda _: reserved,
        set_per_process_memory_fraction=lambda fraction, device: calls.append(("set_fraction", fraction, device)),
    ))
    # Feed a project-local simulated proc tree, never inspect actual OS processes.
    judge._inactive_rank_worker = lambda value: assessment.GPUJudge._inactive_rank_worker(value, tmp_path / "proc")
    return judge, calls, status


def test_capacity_expands_once_in_same_loaded_judge_at_exact_headroom_boundary(tmp_path, monkeypatch):
    judge, calls, _ = cpu_judge(tmp_path, monkeypatch)
    event = judge.maybe_expand_capacity()
    assert event["previous_cap_gib"] == 12 and event["cap_gib"] == 24
    assert event["aggregate_free_bytes"] + event["own_reserved_bytes"] - 24 * GIB == 4 * GIB
    assert judge.prompt_batch_limit == 16 and judge.batch_token_budget == 24000
    assert ("set_fraction", .5, 0) in calls
    assert judge.maybe_expand_capacity() is None
    assert sum(call[0] == "set_fraction" for call in calls) == 1
    assert judge.capacity_upgrade is event and event["at"]


@pytest.mark.parametrize("free,total", [(18 * GIB - 1, 48 * GIB), (100 * GIB, 20 * GIB)])
def test_insufficient_aggregate_headroom_or_device_size_retains_original_cap(tmp_path, monkeypatch, free, total):
    judge, calls, _ = cpu_judge(tmp_path, monkeypatch, free=free, total=total)
    assert judge.maybe_expand_capacity() is None
    assert judge.memory_cap_gib == 12 and judge.prompt_batch_limit == 8
    assert judge.batch_token_budget == 8000 and judge.capacity_upgrade is None
    assert not any(call[0] == "set_fraction" for call in calls)


@pytest.mark.parametrize("status_text", ["broken JSON", "null", '{"state":"scoring"}'])
def test_no_gpu_query_for_incomplete_or_malformed_peer_state(tmp_path, monkeypatch, status_text):
    judge, calls, status = cpu_judge(tmp_path, monkeypatch)
    status.write_text(status_text)
    assert judge.maybe_expand_capacity() is None
    assert calls == []


class LengthTokenizer:
    """Return controlled real token counts; retain every supplied prompt byte."""
    def __init__(self, lengths):
        self.lengths = lengths
        self.calls = []

    def apply_chat_template(self, messages, **options):
        assert options == {"tokenize": False, "add_generation_prompt": True}
        return messages[0]["content"] + "\n" + messages[1]["content"]

    def __call__(self, texts, **options):
        assert options == {"padding": False, "truncation": False}
        self.calls.append(list(texts))
        return {"input_ids": [[0] * self.lengths[next(i for i in range(len(self.lengths)) if f"query-{i}!" in text)]
                              for text in texts]}


def planning_tasks(count):
    packet = {"packet_sha256": "fixed-material", "body": {
        "documented_model_name": "source model", "model_compounds": ["compound"],
        "scope_warning": "frozen warning", "excerpts": [{"n": 1, "text": "complete source evidence"}]}}
    return [(packet, [{"text": f"query-{i}!"}]) for i in range(count)]


@pytest.mark.parametrize("budget,lengths,maximum,expected", [
    (8000, [1000] * 16, 8, 6),
    (24000, [1000] * 16, 16, 16),
    (8000, [1000, 1000, 3000, 1000], 4, 2),
    (24000, [6000] * 16, 16, 3),
    (8000, [10000, 1000], 8, 1),
])
def test_true_padded_token_budget_preserves_order_and_never_truncates(budget, lengths, maximum, expected):
    judge = assessment.GPUJudge.__new__(assessment.GPUJudge)
    judge.tokenizer = LengthTokenizer(lengths)
    judge.batch_token_budget = budget
    judge._prompt_length_cache = {}
    tasks = planning_tasks(len(lengths))
    before = json.dumps(tasks)
    assert judge.select_batch_size(tasks, maximum) == expected
    assert json.dumps(tasks) == before
    assert all("complete source evidence" in text for text in judge.tokenizer.calls[0])
    assert judge.select_batch_size(tasks, maximum) == expected
    assert len(judge.tokenizer.calls) == 1  # Length cache, never a shortened prompt.


def test_oom_reduced_maximum_is_honored_by_next_batch_plan():
    judge = assessment.GPUJudge.__new__(assessment.GPUJudge)
    judge.tokenizer = LengthTokenizer([1000] * 16)
    judge.batch_token_budget = 24000
    judge._prompt_length_cache = {}
    tasks = planning_tasks(16)
    assert judge.select_batch_size(tasks, 16) == 16
    assert judge.select_batch_size(tasks, 4) == 4


def test_token_length_cache_does_not_cross_changed_source_packets():
    judge = assessment.GPUJudge.__new__(assessment.GPUJudge)
    judge.tokenizer = LengthTokenizer([1000])
    judge.batch_token_budget = 8000
    judge._prompt_length_cache = {}
    tasks = planning_tasks(1)
    assert judge.select_batch_size(tasks, 8) == 1
    tasks[0][0]["packet_sha256"] = "other-frozen-material"
    tasks[0][0]["body"]["excerpts"][0]["text"] = "another complete source"
    assert judge.select_batch_size(tasks, 8) == 1
    assert len(judge.tokenizer.calls) == 2
    assert "another complete source" in judge.tokenizer.calls[1][0]
