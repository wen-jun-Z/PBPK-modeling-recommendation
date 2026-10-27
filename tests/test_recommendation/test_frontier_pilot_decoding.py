"""Check the real parser and HF token callback, including zero-item enforcement."""
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import frontier_pilot_decoding as pilot
import frontier_judge_decoding as decoding


def accepts(text, count=1):
    parser = pilot.empty_source_parser(count)
    for c in text:
        if c not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(c)
    return parser.can_end()


@pytest.mark.parametrize("grade", [None, 0, 1, 2, 3])
@pytest.mark.parametrize("applicable", ["Y", "N", "U"])
def test_every_grade_applicability_pair_remains_possible_in_any_field_order(grade, applicable):
    values = {"i": 1, "g": grade, "a": applicable, "e": []}
    for keys in itertools.permutations(values):
        assert accepts(json.dumps([{key: values[key] for key in keys}]))


@pytest.mark.parametrize("evidence", [[0], [1], [999], [-1], [None], [True], ["1"], [[]], [{}]])
def test_every_nonempty_citation_array_is_rejected(evidence):
    assert not accepts(json.dumps([{"i": 1, "g": None, "a": "U", "e": evidence}]))


def test_more_than_one_request_always_has_empty_evidence_without_forcing_labels():
    assert accepts('[{"e":[],"i":1,"g":3,"a":"Y"},{"i":2,"g":0,"a":"N","e":[]}]', 2)
    assert not accepts('[{"e":[],"i":1,"g":3,"a":"Y"},{"i":2,"g":0,"a":"N","e":[1]}]', 2)


def test_immutable_state_and_direct_invalid_character_are_safe():
    parser = pilot.empty_source_parser(1)
    for character in '[{"e":[':
        parser = parser.add_character(character)
    before = parser.get_allowed_characters()
    assert "]" in before and "1" not in before
    with pytest.raises(ValueError, match="empty-evidence grammar"):
        parser.add_character("1")
    after = parser.add_character("]")
    assert parser.get_allowed_characters() == before
    assert after is not parser


@pytest.fixture
def base(monkeypatch):
    from lmformatenforcer.integrations import transformers as integration
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData

    class TinyTokenizer:
        eos_token_id = 128

        def __len__(self):
            return 132

    def build(tokenizer):
        return TokenEnforcerTokenizerData(
            [(value, chr(value), False) for value in range(32, 127)],
            lambda values: "".join(chr(value) for value in values if value < 128),
            tokenizer.eos_token_id, use_bitmask=False, vocab_size=len(tokenizer))

    monkeypatch.setattr(integration, "build_token_enforcer_tokenizer_data", build)
    return decoding.JudgeJSONConstraint(TinyTokenizer(), [128, 129])


def walk(callback, row, text):
    sequence = [130, 130, ord("X")]
    for character in text:
        allowed = callback(row, SimpleNamespace(tolist=lambda: sequence))
        assert ord(character) in allowed
        sequence.append(ord(character))
    return callback(row, SimpleNamespace(tolist=lambda: sequence))


def test_mixed_batch_token_callbacks_preserve_domains_and_official_multi_eos(base):
    constraint = pilot.empty_source_constraint(base)
    tasks = [({"refs": [], "body": {"excerpts": []}}, [{}]), ({"refs": [{}, {}]}, [{}])]
    callback = constraint.for_tasks(tasks)
    prefix = '[{"i":1,"g":3,"a":"Y","e":['
    empty, real = walk(callback, 0, prefix), walk(callback, 1, prefix)
    assert ord("]") in empty and ord("1") not in empty and ord("2") not in empty
    assert ord("1") in real and ord("2") in real
    assert 128 not in empty and 129 not in empty
    complete = walk(callback, 0, prefix + "]}]")
    assert {128, 129}.issubset(complete)


def test_nonempty_packets_keep_original_grammar_and_fresh_calls(base):
    wrapper = pilot.empty_source_constraint(base)
    tasks = [({"refs": [{}, {}]}, [{}])]
    assert ord("2") in walk(wrapper.for_tasks(tasks), 0, '[{"e":[')
    assert ord("2") in walk(wrapper.for_tasks(tasks), 0, '[{"e":[')


def test_empty_malformed_tasks_and_invalid_batch_id_fail(base):
    wrapper = pilot.empty_source_constraint(base)
    with pytest.raises(ValueError, match="empty generation"):
        wrapper.for_tasks([])
    with pytest.raises(ValueError, match="disagree"):
        wrapper.for_tasks([({"refs": [], "body": {"excerpts": [{"text": "present"}]}}, [{}])])
    callback = wrapper.for_tasks([({"refs": []}, [{}])])
    with pytest.raises(ValueError, match="batch row"):
        callback(1, SimpleNamespace(tolist=lambda: []))


def test_original_production_guard_remains_unchanged():
    with pytest.raises(ValueError, match="No supplied excerpts"):
        decoding.judge_json_schema(1, 0)
