"""CPU-only checks of the actual LMFE grammar, with a tiny character vocabulary."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import frontier_judge_decoding as decoding
import run_frontier_assessment as assessment


lmfe = pytest.importorskip("lmformatenforcer")


def grammar_accepts(text, request_count=1, reference_count=3):
    parser = lmfe.JsonSchemaParser(decoding.judge_json_schema(request_count, reference_count))
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


@pytest.mark.parametrize("grade", [None, 0, 1, 2, 3])
@pytest.mark.parametrize("applicability", ["Y", "N", "U"])
def test_grammar_preserves_all_judgment_options(grade, applicability):
    answer = [{"i": 1, "g": grade, "a": applicability, "e": [1]}]
    assert grammar_accepts(json.dumps(answer))


def test_unknown_empty_refs_and_reordered_fields_remain_allowed():
    assert grammar_accepts('[{"a":"U","e":[],"g":null,"i":1}]')


@pytest.mark.parametrize("invalid", [
    '[{"i":1,"g":4,"a":"Y","e":[1]}]',
    '[{"i":1,"g":true,"a":"Y","e":[1]}]',
    '[{"i":1,"g":1.0,"a":"Y","e":[1]}]',
    '[{"i":1,"g":null,"a":"Maybe","e":[]}]',
    '[{"i":2,"g":null,"a":"U","e":[]}]',
    '[{"i":1,"g":null,"a":"U","e":[4]}]',
    '[{"i":1,"g":null,"a":"U","e":[0]}]',
    '[{"i":1,"g":null,"a":"U","e":[true]}]',
    '[{"i":1,"g":null,"a":"U","e":[1,2,3,1]}]',
    '[{"i":1,"g":null,"a":"U","e":[],"why":"extra"}]',
    '[{"i":1,"g":null,"a":"U"}]',
    '[]',
    '[{"i":1,"g":null,"a":"U","e":[]},{"i":1,"g":null,"a":"U","e":[]}]',
    '[{"i":1,"g":null,"a":"U","e":[]}] Here is why.',
    '```json\n[{"i":1,"g":null,"a":"U","e":[]}]\n```',
])
def test_grammar_blocks_invalid_json_fields_types_ids_and_explanations(invalid):
    assert not grammar_accepts(invalid)


@pytest.mark.parametrize("answer", [
    [{"i": 1, "g": 3, "a": "Y", "e": []}],
    [{"i": 1, "g": 3, "a": "Y", "e": [1, 1]}],
])
def test_semantic_checks_still_reject_grammar_valid_answers(answer):
    raw = json.dumps(answer)
    assert grammar_accepts(raw)
    with pytest.raises(ValueError):
        assessment.parse_answers(raw, [1], [{"source_id": "real"}] * 3)


def test_duplicate_request_ids_remain_post_decode_validation():
    raw = json.dumps([{"i": 1, "g": None, "a": "U", "e": []}] * 2)
    assert grammar_accepts(raw, request_count=2)
    with pytest.raises(ValueError, match="repeated request"):
        assessment.parse_answers(raw, [1, 2], [{}] * 3)


@pytest.mark.parametrize("counts", [(0, 1), (True, 1), (1, 0), (1, False), (1, -1)])
def test_invalid_sizes_do_not_start_decoding(counts):
    with pytest.raises(ValueError):
        decoding.judge_json_schema(*counts)


class TinyTokenizer:
    eos_token_id = 128

    def __len__(self):
        return 132


@pytest.fixture
def tiny_constraint(monkeypatch):
    from lmformatenforcer.integrations import transformers as integration
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData

    builds = []

    def build(tokenizer):
        builds.append(tokenizer)
        return TokenEnforcerTokenizerData(
            [(value, chr(value), False) for value in range(32, 127)],
            lambda values: "".join(chr(value) for value in values if value < 128),
            tokenizer.eos_token_id,
            use_bitmask=False,
            vocab_size=len(tokenizer),
        )

    monkeypatch.setattr(integration, "build_token_enforcer_tokenizer_data", build)
    constraint = decoding.JudgeJSONConstraint(TinyTokenizer(), [128, 129])
    return constraint, builds


def tensor(sequence):
    return SimpleNamespace(tolist=lambda: list(sequence))


def walk(callback, batch_id, text, prompt=None):
    # A padded prompt containing arbitrary non-JSON characters must not be parsed.
    sequence = list(prompt or [130, 130, ord("X")])
    for character in text:
        allowed = callback(batch_id, tensor(sequence))
        assert ord(character) in allowed, (text, character, allowed)
        sequence.append(ord(character))
    return sequence, callback(batch_id, tensor(sequence))


def test_token_filter_preserves_model_multi_eos_and_disallows_early_stop(tiny_constraint):
    constraint, _ = tiny_constraint
    callback = constraint.for_tasks([({"refs": [{}]}, [{}])])
    _, before = walk(callback, 0, '[{"i":1,"g":null,"a":"U","e":[]')
    assert 128 not in before and 129 not in before
    complete, after = walk(callback, 0, '[{"i":1,"g":null,"a":"U","e":[]}]')
    assert {128, 129}.issubset(after)
    assert ord("H") not in after  # No explanation after the array.
    # A finished batch row remains safe while a different row is still decoding.
    complete.append(129)
    assert callback(0, tensor(complete))
    complete.append(131)
    assert callback(0, tensor(complete))


def test_per_row_reference_enums_with_identical_left_padded_prompts(tiny_constraint):
    constraint, _ = tiny_constraint
    callback = constraint.for_tasks([
        ({"refs": [{}]}, [{}]),
        ({"refs": [{}] * 12}, [{}]),
    ])
    prefix = '[{"i":1,"g":0,"a":"N","e":['
    _, first = walk(callback, 0, prefix)
    _, second = walk(callback, 1, prefix)
    assert ord("2") not in first and ord("2") in second
    assert ord("]") in first and ord("]") in second  # Never force citation.
    _, after_twelve = walk(callback, 1, prefix + "12")
    assert ord("]") in after_twelve and ord("3") not in after_twelve


def test_tokenizer_built_once_but_generation_parsers_are_fresh(tiny_constraint):
    constraint, builds = tiny_constraint
    tasks = [({"refs": [{}]}, [{}])]
    first = constraint.for_tasks(tasks)
    walk(first, 0, '[{"i":1,"g":null,"a":"U","e":[]}]')
    second = constraint.for_tasks(tasks)
    # Deliberately use the previous completed sequence as the new prompt. A reused
    # enforcer would consider it the prior JSON's end and wrongly allow EOS.
    prior_complete = [130, 130, ord("X")] + list(map(ord, '[{"i":1,"g":null,"a":"U","e":[]}]'))
    allowed = second(0, tensor(prior_complete))
    assert ord("[") in allowed and 128 not in allowed and 129 not in allowed
    assert len(builds) == 1


def test_every_grade_and_unknown_branch_is_available_to_model(tiny_constraint):
    constraint, _ = tiny_constraint
    callback = constraint.for_tasks([({"refs": [{}]}, [{}])])
    _, allowed = walk(callback, 0, '[{"i":1,"g":')
    assert set(map(ord, "0123n")).issubset(allowed)
    assert ord("4") not in allowed and ord("t") not in allowed


def test_empty_and_foreign_batch_rejected(tiny_constraint):
    constraint, _ = tiny_constraint
    with pytest.raises(ValueError, match="empty generation"):
        constraint.for_tasks([])
    callback = constraint.for_tasks([({"refs": [{}]}, [{}])])
    with pytest.raises(ValueError, match="Unknown generation"):
        callback(1, tensor([130]))


def test_protocol_discloses_syntax_only_and_post_validation():
    protocol = decoding.decoding_protocol()
    assert protocol["semantic_validation_required"] is True
    assert protocol["semantic_correctness_guaranteed"] is False
    assert protocol["token_limit_may_truncate"] is True
    assert len(protocol["schema_template_sha256"]) == 64
    assert protocol["library_version"]
