"""Contracts for the new offline evidence endpoint; no GPU/model load."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("evidence_qwen_server", ROOT / "scripts/deployment/evidence_qwen_server.py")
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def body():
    return {"model": server.MODEL,
            "messages": [{"role": "system", "content": "Quote only supplied evidence."},
                         {"role": "user", "content": "Source: clearance was 3 L/h."}],
            "temperature": 0, "max_tokens": 128,
            "response_format": {"type": "json_schema", "json_schema": {
                "schema": {"type": "object", "required": ["value"],
                           "properties": {"value": {"type": "number"}}}}}}


def test_accepts_explicit_schema_and_preserves_sources():
    request = body()
    original = copy.deepcopy(request)
    assert server.validate_request(request) == original


@pytest.mark.parametrize("change", [
    {"model": "copd-primary"}, {"max_tokens": 4097},
    {"messages": [{"role": "tool", "content": "invoke other service"}]},
    {"response_format": {"type": "json_object"}},
    {"temperature": .7}, {"chat_template_kwargs": {"enable_thinking": True}},
])
def test_rejects_unregistered_generation(change):
    request = body()
    request.update(change)
    with pytest.raises(ValueError):
        server.validate_request(request)


def test_seeded_recovery_cannot_join_greedy_batch():
    request = body()
    request.update(temperature=.7, top_p=.8, top_k=20, min_p=0.0, seed=20260908)
    server.validate_request(request)
    assert server.generation_group(request) is None
    assert server.generation_group(body()) == (128, 1.0)


def test_generation_completion_requires_eos_and_counts_each_row_without_padding():
    assert server.generation_outcome([42, 99, 0, 0], {99}, 4) == ("stop", 2, "eos")
    assert server.generation_outcome([42, 43], {99}, 4) == ("length", 2, "generation_time_limit")
    assert server.generation_outcome([42, 43, 44, 45], {99}, 4) == ("length", 4, "output_token_limit")


def test_single_maximal_request_fits_token_budget():
    assert server.MAX_INPUT + server.MAX_OUTPUT <= server.TOKEN_BUDGET


def test_invalid_json_preserves_raw_and_position_without_discarding_valid_rows():
    task = {"body": body(), "length": 31}
    raw = ['{"value": 3}', '{"value": "bad\tcontrol"}', '{"value": 4}']
    rows = [server.generation_result(task, content, [42, 99], {99}, 128) for content in raw]
    assert [row["choices"][0]["finish_reason"] for row in rows] == ["stop", "invalid_json", "stop"]
    assert rows[1]["choices"][0]["message"]["content"] == raw[1]
    assert rows[1]["generation"]["error_type"] == "JSONDecodeError"
    assert rows[1]["generation"]["error_position"] == {"offset": 14, "line": 1, "column": 15}
    assert rows[1]["usage"] == {"prompt_tokens": 31, "completion_tokens": 2}
    assert rows[0]["generation"]["stop_reason"] == "eos"


def test_invalid_schema_keeps_raw_and_does_not_weaken_schema():
    task = {"body": body(), "length": 31}
    original = copy.deepcopy(task)
    content = '{"value": "not a number"}'
    row = server.generation_result(task, content, [42, 99], {99}, 128)
    assert row["choices"][0]["finish_reason"] == "invalid_schema"
    assert row["choices"][0]["message"]["content"] == content
    assert row["generation"]["error_type"] == "ValidationError"
    assert row["generation"]["error_path"] == ["value"]
    assert task == original


def test_generation_counts_distinguishes_invalid_output_from_truncation():
    task = {"body": body(), "length": 31}
    rows = [server.generation_result(task, content, tokens, {99}, 128)
            for content, tokens in [('{"value": 3}', [42, 99]),
                                    ('{"value":', [42]),
                                    ('{"value": }', [42, 99]),
                                    ('{"value": "bad"}', [42, 99])]]
    assert rows[1]["choices"][0]["finish_reason"] == "length"
    assert rows[1]["generation"]["stop_reason"] == "generation_time_limit"
    assert server.generation_counts(rows) == {
        "responded_requests": 4, "completed_requests": 1, "truncated_requests": 1,
        "invalid_requests": 2, "failed_requests": 3,
    }
