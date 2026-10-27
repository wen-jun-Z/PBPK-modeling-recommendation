"""Independent CPU checks of two-stage token continuity and recorded provenance.

Uses the project's pinned official tokenizer only; no model weights or CUDA.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HOME=str(ROOT / "runtime/hf_cache/frontier"),
                  HF_HUB_CACHE=str(ROOT / "runtime/hf_cache/frontier/hub"),
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, str(ROOT / "scripts"))

import frontier_reasoned_judge as reasoned
import frontier_judge_prompt_v5 as prompt_v5
import frontier_modern_judge as modern
import run_frontier_assessment as assessment


@pytest.fixture(scope="module")
def official():
    from transformers import AutoTokenizer

    manifest = json.loads(modern.WEIGHTS.read_text())
    path = Path(manifest["path"])
    assert path.resolve().is_relative_to(ROOT)
    assert manifest["repo"] == modern.REPO and manifest["revision"] == modern.REVISION
    for name in ("tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == manifest["sha256"][name]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    generation = json.loads((path / "generation_config.json").read_text())
    return SimpleNamespace(tokenizer=tokenizer, weights=manifest,
                           eos=generation["eos_token_id"], pad=generation["pad_token_id"],
                           think_end=tokenizer.convert_tokens_to_ids("</think>"),
                           separator=tokenizer.encode("\n\n", add_special_tokens=False))


def test_official_native_thinking_template_and_eos_are_pinned(official):
    t = official.tokenizer
    messages = [{"role": "system", "content": "Frozen system."},
                {"role": "user", "content": "Read the source."}]
    native = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    disabled = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert native.endswith("<|im_start|>assistant\n<think>\n")
    assert disabled.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert official.eos == [248046, 248044]
    assert official.pad == 248044 and official.pad in official.eos
    assert t.encode("</think>", add_special_tokens=False) == [official.think_end]
    assert official.think_end not in official.eos


def test_transformers_eos_stopping_is_per_row_and_preserves_unfinished_rows(official):
    import torch
    from transformers.generation.stopping_criteria import EosTokenCriteria

    criterion = EosTokenCriteria(eos_token_id=[official.think_end, *official.eos])
    token = official.tokenizer.encode("x", add_special_tokens=False)[0]
    rows = torch.tensor([[token, official.think_end], [token, official.eos[0]],
                         [token, official.eos[1]], [token, token]])
    stopped = criterion(rows, scores=None)
    assert stopped.tolist() == [True, True, True, False]
    assert (~stopped).tolist() == [False, False, False, True]


def test_natural_stop_cuts_only_batched_padding(official):
    analysis = official.tokenizer.encode("Synthetic analysis token fixture.", add_special_tokens=False)
    value = reasoned.split_reasoning_tokens(analysis + [official.think_end] + [official.pad] * 7,
                                           official.think_end, official.eos, official.pad)
    assert value["analysis_token_ids"] == analysis
    assert value["raw_token_ids"] == analysis + [official.think_end]
    assert value["stop_reason"] == "think_end" and value["stop_token_id"] == official.think_end
    assert value["native_thinking_completed"] is True
    assert value["forced_think_closure"] is False and value["budget_exhausted"] is False
    assert value["padding_tokens_discarded"] == 7


@pytest.mark.parametrize("eos_index", [0, 1])
def test_official_eos_including_pad_alias_is_not_mistaken_for_budget(official, eos_index):
    analysis = official.tokenizer.encode("Short synthetic analysis.", add_special_tokens=False)
    stop = official.eos[eos_index]
    value = reasoned.split_reasoning_tokens(analysis + [stop] + [official.pad] * 4,
                                           official.think_end, official.eos, official.pad)
    assert value["analysis_token_ids"] == analysis
    assert value["raw_token_ids"] == analysis + [stop]
    assert value["stop_reason"] == "official_eos" and value["stop_token_id"] == stop
    assert value["native_thinking_completed"] is False and value["forced_think_closure"] is True
    assert value["budget_exhausted"] is False and value["padding_tokens_discarded"] == 4


def test_budget_exhaustion_remains_distinct_from_unexpected_early_stop(official):
    token = official.tokenizer.encode("x", add_special_tokens=False)[0]
    full = reasoned.split_reasoning_tokens([token] * 1024, official.think_end, official.eos, official.pad)
    early = reasoned.split_reasoning_tokens([token] * 7, official.think_end, official.eos, official.pad)
    assert full["analysis_token_ids"] == [token] * 1024
    assert full["stop_reason"] == "budget_exhausted" and full["budget_exhausted"] is True
    assert early["stop_reason"] == "unexpected_early_stop" and early["budget_exhausted"] is False
    assert full["forced_think_closure"] and early["forced_think_closure"]


def test_nonpadding_generation_after_stop_is_never_silently_discarded(official):
    token = official.tokenizer.encode("x", add_special_tokens=False)[0]
    for stop in [official.think_end, *official.eos]:
        with pytest.raises(ValueError):
            reasoned.split_reasoning_tokens([token, stop, token], official.think_end, official.eos, official.pad)


@pytest.mark.parametrize("stop_kind", ["native", "eos", "budget"])
def test_continuation_preserves_every_original_input_and_actual_analysis_token(official, stop_kind):
    t = official.tokenizer
    source = "Keep exact Unicode β and spacing:  x\n[7] untouched evidence."
    prompt = t.apply_chat_template([{"role": "user", "content": source}],
                                  tokenize=True, add_generation_prompt=True, enable_thinking=True)
    actual = t.encode("Synthetic token-preservation fixture.", add_special_tokens=False)
    if stop_kind == "native": generated = actual + [official.think_end]
    elif stop_kind == "eos": generated = actual + [official.eos[0]]
    else: generated = [actual[0]] * 1024
    record = reasoned.split_reasoning_tokens(generated, official.think_end, official.eos, official.pad)
    combined = reasoned.continuation_tokens(prompt, record, official.think_end, official.separator)
    assert combined == list(prompt) + record["analysis_token_ids"] + [official.think_end] + official.separator
    assert combined[:len(prompt)] == list(prompt)
    assert combined[len(prompt):len(prompt) + len(record["analysis_token_ids"])] == record["analysis_token_ids"]


def test_left_padding_retains_row_identity_and_all_tokens(official):
    import torch

    rows = [[11, 12], [21, 22, 23, 24], [31]]
    values = reasoned.pad_token_rows(rows, official.pad, torch)
    assert values["input_ids"].device.type == "cpu"
    assert values["input_ids"].shape == values["attention_mask"].shape == (3, 4)
    for row, mask, expected in zip(values["input_ids"].tolist(), values["attention_mask"].tolist(), rows):
        assert [token for token, keep in zip(row, mask) if keep] == expected
        assert mask == [0] * (4 - len(expected)) + [1] * len(expected)


def _tasks():
    result = []
    for i, text in enumerate(("First source body.", "Second source body with extra Unicode β and two  spaces.")):
        body = {"documented_model_name": f"fixture-{i}", "model_compounds": [f"fixture-drug-{i}"],
                "excerpts": [{"n": 1, "text": text}], "scope_warning": "CPU fixture only"}
        packet = {"body": body, "refs": [{"quote": text, "source_id": f"fixture-source-{i}"}],
                  "packet_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                  "source_candidate_id": f"fixture-{i}"}
        query = {"query_id": f"fixture-q-{i}", "text": f"Find a documented PBPK model for fixture-drug-{i}.",
                 "task_type": "plasma_concentration_time_prediction", "compounds": [f"fixture-drug-{i}"], "features": {}}
        result.append((packet, [query]))
    return result


def _read_artifact(pointer):
    path = Path(pointer["path"])
    if not path.is_absolute(): path = ROOT / path
    assert path.resolve().is_relative_to(ROOT)
    chunk = path.read_bytes()[pointer["offset"]:pointer["offset"] + pointer["length"]]
    assert hashlib.sha256(chunk).hexdigest() == pointer["sha256"]
    assert pointer["task_count"] == 2
    return path, json.loads(chunk)


class _ConstraintSpy:
    def __init__(self, eos):
        self.seen = []
        self.eos = eos

    def for_tasks(self, tasks):
        self.seen.append(tasks)
        return lambda batch_id, token_ids: [self.eos]


class _CPUModel:
    def __init__(self, official, torch):
        self.official = official
        self.torch = torch
        self.calls = []
        self.fail_answer = False
        self.corrupt_cardinality = False
        self.corrupt_prefix = False

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append({"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone(), **kwargs})
        stage = (len(self.calls) - 1) % 2
        if stage == 1 and self.fail_answer:
            raise RuntimeError("Synthetic second-phase failure")
        assert input_ids.device.type == "cpu"
        assert kwargs["do_sample"] is False
        if stage == 0:
            token = self.official.tokenizer.encode("x", add_special_tokens=False)[0]
            rows = [[token, token, self.official.think_end], [token] * 1024]
            assert kwargs["max_new_tokens"] == 1024
            assert kwargs["eos_token_id"] == [self.official.think_end, *self.official.eos]
            assert "prefix_allowed_tokens_fn" not in kwargs
        else:
            assert kwargs["eos_token_id"] == self.official.eos
            assert callable(kwargs["prefix_allowed_tokens_fn"])
            rows = [self.official.tokenizer.encode('[{"i":1,"g":null,"a":"U","e":[]}]', add_special_tokens=False) + [self.official.eos[0]]
                    for _ in range(input_ids.shape[0])]
        width = max(map(len, rows))
        generated = self.torch.tensor([row + [self.official.pad] * (width - len(row)) for row in rows], dtype=self.torch.long)
        output = self.torch.cat([input_ids.clone(), generated], dim=1)
        if self.corrupt_cardinality: output = output[:1]
        if self.corrupt_prefix: output[0, 0] = output[0, 0] + 1
        return output


@pytest.fixture
def cpu_judge(official, tmp_path, monkeypatch):
    import torch

    assert tmp_path.resolve().is_relative_to(ROOT), "Run pytest with a project-local --basetemp"
    monkeypatch.setattr(assessment, "ASSESS", tmp_path)
    monkeypatch.setattr(assessment, "SYSTEM", prompt_v5.SYSTEM)
    (tmp_path / "judge_protocol.json").write_text('{"profile":"synthetic-cpu-test"}\n')
    # No model constructor is called: only an actual CPU Tensor fake is used.
    judge = reasoned.ReasonedGPUJudge.__new__(reasoned.ReasonedGPUJudge)
    judge.torch = torch
    judge.device = "cpu"
    judge.tokenizer = official.tokenizer
    judge.model = _CPUModel(official, torch)
    judge.json_constraint = _ConstraintSpy(official.eos[0])
    judge.weights = official.weights
    judge.think_end_id = official.think_end
    judge.official_eos = list(official.eos)
    judge.separator_ids = list(official.separator)
    judge.reasoning_tokens = 1024
    judge.memory_cap_gib = 36
    judge.prompt_batch_limit = 16
    judge.batch_token_budget = 32000
    judge.capacity_upgrade = None
    judge._prompt_length_cache = {}
    judge.metadata = {}
    # This makes the real disk guard independent of filesystem occupancy in a unit fixture.
    monkeypatch.setattr(reasoned.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    return judge


def _unpadded(call):
    return [[token for token, keep in zip(row, mask) if keep]
            for row, mask in zip(call["input_ids"].tolist(), call["attention_mask"].tolist())]


def test_mocked_two_phase_generation_preserves_native_input_and_records_real_tokens(cpu_judge, official):
    tasks = _tasks()
    expected_input = [official.tokenizer.encode(cpu_judge._render(packet, queries), add_special_tokens=False)
                      for packet, queries in tasks]
    values, metrics = cpu_judge.generate(tasks)
    assert len(values) == 2 and all(json.loads(text)[0]["g"] is None for text in values)
    first, second = cpu_judge.model.calls
    assert _unpadded(first) == expected_input
    assert second["max_new_tokens"] == 96
    path, thinking = _read_artifact(metrics["reasoning_artifact"])
    _, final = _read_artifact(metrics["final_artifact"])
    assert thinking["phase"] == "thinking" and final["phase"] == "answer"
    assert len(path.read_text().splitlines()) == 2
    assert thinking["attempt_id"] == final["attempt_id"]
    assert final["reasoning_artifact"] == metrics["reasoning_artifact"]
    assert metrics["final_artifact"]["offset"] == metrics["reasoning_artifact"]["length"]
    assert thinking["reasoning"][0]["stop_reason"] == "think_end"
    assert thinking["reasoning"][1]["stop_reason"] == "budget_exhausted"
    assert metrics["forced_think_closure"] == [False, True]
    reconstructed = [{**item, "analysis_token_ids": item["raw_token_ids"][:item["analysis_token_count"]]}
                     for item in thinking["reasoning"]]
    assert all("analysis_token_ids" not in item and "analysis_text" not in item
               for item in thinking["reasoning"])
    expected_final = [reasoned.continuation_tokens(prompt, item, official.think_end, official.separator)
                      for prompt, item in zip(expected_input, reconstructed)]
    assert _unpadded(second) == expected_final
    for i, (packet, queries) in enumerate(tasks):
        context = thinking["tasks"][i]
        assert context["packet_sha256"] == packet["packet_sha256"]
        assert context["query_ids"] == [queries[0]["query_id"]]
        assert context["input_token_ids_sha256"] == reasoned.sha(reasoned.canonical(expected_input[i]))
        assert final["final_input_token_ids_sha256"][i] == reasoned.sha(reasoned.canonical(expected_final[i]))
    assert cpu_judge.json_constraint.seen == [tasks]
    assert "analysis_text" not in json.dumps(metrics) and "raw_token_ids" not in json.dumps(metrics)


def test_repair_remains_source_bound_new_attempt_and_uses_160_token_budget(cpu_judge):
    tasks = _tasks()
    _, first_metrics = cpu_judge.generate(tasks)
    path, first = _read_artifact(first_metrics["reasoning_artifact"])
    preserved = path.read_bytes()
    repairs = [{"output": '[{"i":1,"g":3,"a":"Y","e":[]}]', "error": "Known answer missing citations"}
               for _ in tasks]
    _, second_metrics = cpu_judge.generate(tasks, max_new_tokens=160, repairs=repairs)
    _, second = _read_artifact(second_metrics["reasoning_artifact"])
    assert path.read_bytes().startswith(preserved)
    assert len(path.read_text().splitlines()) == 4
    assert first["attempt_id"] != second["attempt_id"]
    assert cpu_judge.model.calls[-1]["max_new_tokens"] == 160
    for i in range(2):
        assert first["tasks"][i]["packet_body_sha256"] == second["tasks"][i]["packet_body_sha256"]
        assert first["tasks"][i]["source_refs_sha256"] == second["tasks"][i]["source_refs_sha256"]
        assert second["tasks"][i]["repair"] is True
        assert second["tasks"][i]["repair_payload_sha256"] == reasoned.sha(reasoned.canonical(repairs[i]))
        assert second["tasks"][i]["rendered_prompt_sha256"] != first["tasks"][i]["rendered_prompt_sha256"]
        rendered = cpu_judge._render(*tasks[i], repair=repairs[i])
        assert tasks[i][0]["body"]["excerpts"][0]["text"] in rendered
        assert tasks[i][1][0]["text"] in rendered
        assert repairs[i]["error"] in rendered


def test_phase_two_failure_preserves_the_already_committed_reasoning(cpu_judge):
    cpu_judge.model.fail_answer = True
    with pytest.raises(RuntimeError, match="Synthetic second-phase failure"):
        cpu_judge.generate(_tasks())
    path = assessment.ASSESS / "reasoning/qwen_generation_journal.jsonl"
    records = list(map(json.loads, path.read_text().splitlines()))
    assert [r["phase"] for r in records] == ["thinking", "answer_error"]
    assert records[0]["attempt_id"] == records[1]["attempt_id"]
    _, preserved = _read_artifact(records[1]["reasoning_artifact"])
    assert preserved == records[0]
    assert records[1]["error_type"] == "RuntimeError"


@pytest.mark.parametrize("fault", ["corrupt_cardinality", "corrupt_prefix"])
def test_model_cannot_silently_drop_a_task_or_modify_input(cpu_judge, fault):
    setattr(cpu_judge.model, fault, True)
    with pytest.raises(ValueError, match="cardinality"):
        cpu_judge.generate(_tasks())


def test_empty_or_mismatched_repairs_rejected_before_any_inference(cpu_judge):
    with pytest.raises(ValueError): cpu_judge.generate([])
    with pytest.raises(ValueError): cpu_judge.generate(_tasks(), repairs=[{}])
    with pytest.raises(ValueError): cpu_judge.generate(_tasks(), max_new_tokens=161)
    with pytest.raises(ValueError): cpu_judge.generate(_tasks() * 9)
    assert cpu_judge.model.calls == []


def test_frozen_protocol_discloses_greedy_bounded_thinking_and_forced_closure(official):
    protocol = reasoned.generation_protocol(official.weights)
    assert protocol["repo"] == modern.REPO and protocol["revision"] == modern.REVISION
    assert protocol["phase1"]["enable_thinking"] and protocol["phase1"]["max_new_tokens"] == 1024
    assert protocol["phase1"]["json_constraint"] is False
    assert protocol["phase1"]["per_row_stop_ids"] == [official.think_end, *official.eos]
    assert protocol["phase2"]["official_eos_ids"] == official.eos
    assert protocol["phase2"]["repair_max_new_tokens"] == 160
    assert protocol["semantic_entailment_verified"] is False
    assert protocol["journal"]["phase1_written_before_phase2"] is True
