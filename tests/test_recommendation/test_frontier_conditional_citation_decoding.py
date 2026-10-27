"""CPU contract and pinned official Phi-tokenizer checks; no model inference."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import frontier_conditional_citation_decoding as decoding
import frontier_judge_decoding as original
import run_frontier_assessment as assessment


def advance(parser, text):
    for character in text:
        if character not in parser.get_allowed_characters():
            raise ValueError((text, character))
        parser = parser.add_character(character)
    return parser


def accepts(text, refs=3, requests=1):
    try:
        return advance(decoding.conditional_citation_parser(requests, refs), text).can_end()
    except ValueError:
        return False


@pytest.mark.parametrize("grade,app", itertools.product([None, 0, 1, 2, 3], ["Y", "N", "U"]))
def test_every_valid_answer_survives_in_grade_first_order(grade, app):
    for count in range(1, 4):
        for refs in itertools.permutations([1, 2, 3], count):
            output = json.dumps([dict(i=1, g=grade, a=app, e=list(refs))])
            assert accepts(output)
            value = assessment.parse_answers(output, [1], [{}, {}, {}])[1]
            assert value["grade"] == grade and value["evidence_numbers"] == list(refs)
    assert accepts(json.dumps([dict(i=1, g=grade, a=app, e=[])])) is (grade is None and app == "U")


@pytest.mark.parametrize("grade,app", itertools.product([None, 0, 1, 2, 3], ["Y", "N", "U"]))
def test_decisions_are_chosen_before_evidence_but_known_cannot_close_empty(grade, app):
    prefix = json.dumps([dict(i=1, g=grade, a=app, e=[])], separators=(",", ":"))[:-3]
    parser = advance(decoding.conditional_citation_parser(1, 12), prefix)
    chars = parser.get_allowed_characters()
    assert "1" in chars and "2" in chars
    assert ("]" in chars) is (grade is None and app == "U")
    assert decoding.FIELD_ORDER == ("i", "g", "a", "e")


def test_empty_document_allows_only_unknown_without_any_reference():
    for grade, app in itertools.product([None, 0, 1, 2, 3], ["Y", "N", "U"]):
        assert accepts(json.dumps([dict(i=1, g=grade, a=app, e=[])]), refs=0) is (grade is None and app == "U")
    assert not accepts('[{"i":1,"g":null,"a":"U","e":[1]}]', refs=0)


@pytest.mark.parametrize("evidence", [[0], [-1], [4], [1, 1], [1, 2, 1], [1, 2, 3, 4], [1.0], [True], ["1"]])
def test_invalid_reference_values_rejected(evidence):
    assert not accepts(json.dumps([dict(i=1, g=0, a="N", e=evidence)]))


@pytest.mark.parametrize("raw", [
    '[{"i":1,"g":0,"a":"N","e":[1,]}]',
    '[{"i":1,"g":0,"a":"N","e":[1e0]}]',
    '[{"i":1,"g":0,"a":"N","e":[01]}]',
    '[{"i":1,"e":[1],"g":0,"a":"N"}]',
    '[{"i":1,"a":"N","g":0,"e":[1]}]',
    '[{"i":1,"g":4,"a":"N","e":[1]}]',
    '[{"i":1,"g":0,"a":"Maybe","e":[1]}]',
    '[{"i":1,"g":0,"a":"N","e":[1],"reason":"x"}]',
    '[{"i":1,"g":0,"a":"N","\\u0065":[]}]',
    '[{"i":1,"\\u0067":0,"a":"N","e":[]}]',
    '[{"i":1,"g":0,"a":"\\u004e","e":[]}]',
    '[{"i":1,"g":null,"a":"U","e":[]}] explanation',
    '[]',
])
def test_invalid_order_escaped_bypass_and_schema_rejected(raw):
    assert not accepts(raw)


def test_multidigit_references_sharing_a_prefix_and_third_citation():
    parser = advance(decoding.conditional_citation_parser(1, 12), '[{"i":1,"g":0,"a":"N","e":[1,1')
    chars = parser.get_allowed_characters()
    assert "]" not in chars and "," not in chars
    assert set("012") <= set(chars)
    for refs in ([1, 10], [1, 11], [1, 12], [12, 1], [12, 10, 11]):
        assert accepts(json.dumps([dict(i=1, g=0, a="N", e=refs)]), refs=12)
    assert not accepts('[{"i":1,"g":0,"a":"N","e":[12,12]}]', refs=12)


def test_cross_object_state_is_not_inherited_and_request_order_stays_free():
    rows = [dict(i=2, g=0, a="N", e=[1]), dict(i=1, g=None, a="U", e=[]),
            dict(i=3, g=None, a="Y", e=[2])]
    raw = json.dumps(rows)
    assert accepts(raw, refs=2, requests=3)
    assert set(assessment.parse_answers(raw, [1, 2, 3], [{}, {}])) == {1, 2, 3}
    assert not accepts(raw, refs=2, requests=2)
    rows[-1]["e"] = []
    assert not accepts(json.dumps(rows), refs=2, requests=3)


def test_immutable_prefix_and_bounded_whitespace():
    start = decoding.conditional_citation_parser(1, 3)
    branch = advance(start, '[{"i":1,"g":0,"a":"N","e":[')
    assert "[" in start.get_allowed_characters() and not start.can_end()
    assert "]" not in branch.get_allowed_characters()
    assert accepts('[\n { "i" : 1, "g" : 0, "a" : "N", "e" : [ 1, 2 ] }]')
    assert not accepts('[   {"i":1,"g":null,"a":"U","e":[]}]')


@pytest.mark.parametrize("counts", [(0, 1), (True, 1), (1, True), (1, -1), (1, 1.0)])
def test_invalid_counts_rejected(counts):
    with pytest.raises(ValueError): decoding.conditional_citation_parser(*counts)


@pytest.fixture(scope="module")
def phi():
    from transformers import AutoTokenizer
    manifest = json.loads((ROOT / "experiments/model_recommendation/frontier_comparison_20260914/assessment/primary_weights.json").read_text())
    assert manifest["repo"] == "microsoft/phi-4"
    path = Path(manifest["path"])
    assert path.resolve().is_relative_to(ROOT)
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == manifest["sha256"][name]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    eos = json.loads((path / "generation_config.json").read_text())["eos_token_id"]
    base = original.JudgeJSONConstraint(tokenizer, eos)
    return tokenizer, base, decoding.conditional_citation_constraint(base)


def token_walk(phi, callback, row, output, prompt=None):
    import torch
    tokenizer = phi[0]
    ids = list(prompt if prompt is not None else tokenizer.encode("Private same prompt.", add_special_tokens=False))
    for token in tokenizer.encode(output, add_special_tokens=False):
        allowed = callback(row, torch.tensor(ids, dtype=torch.long))
        assert token in allowed, (output, tokenizer.decode([token]))
        ids.append(token)
    return ids, callback(row, torch.tensor(ids, dtype=torch.long))


@pytest.mark.parametrize("grade,app", itertools.product([None, 0, 1, 2, 3], ["Y", "N", "U"]))
def test_real_phi_tokens_preserve_every_nonempty_decision_and_official_eos(phi, grade, app):
    callback = phi[2].for_tasks([({"refs": [{}] * 12}, [{}])])
    output = json.dumps([dict(i=1, g=grade, a=app, e=[1, 12, 10])], separators=(",", ":"))
    _, allowed = token_walk(phi, callback, 0, output)
    assert set(phi[1].eos_token_ids) <= set(allowed)


@pytest.mark.parametrize("raw", [
    '[{"i":1,"g":0,"a":"N","e":[]}]',
    '[{"i":1,"g":null,"a":"Y","e":[]}]',
    '[{"i":1,"g":2,"a":"U","e":[]}]',
    '[{"i":1,"g":0,"a":"N","e":[1,1]}]',
    '[{"i":1,"g":0,"a":"N","e":[13]}]',
    '[{"i":1,"e":[1],"g":0,"a":"N"}]',
])
def test_real_phi_tokens_cannot_bypass_the_contract(phi, raw):
    callback = phi[2].for_tasks([({"refs": [{}] * 12}, [{}])])
    with pytest.raises(AssertionError): token_walk(phi, callback, 0, raw)


def test_real_phi_rows_calls_and_eos_remain_isolated(phi):
    import torch
    tokenizer, base, wrapped = phi
    assert wrapped.tokenizer_data is base.tokenizer_data and wrapped.eos_token_ids is base.eos_token_ids
    assert decoding.conditional_citation_constraint(wrapped) is wrapped
    callback = wrapped.for_tasks([({"refs": [], "body": {"excerpts": []}}, [{}]), ({"refs": [{}] * 12}, [{}])])
    empty = '[{"i":1,"g":null,"a":"U","e":[]}]'
    cited = '[{"i":1,"g":0,"a":"N","e":[12]}]'
    first, allowed = token_walk(phi, callback, 0, empty)
    assert set(base.eos_token_ids) <= set(allowed)
    _, allowed = token_walk(phi, callback, 1, cited)
    assert set(base.eos_token_ids) <= set(allowed)
    second = wrapped.for_tasks([({"refs": [{}]}, [{}])])
    _, partial = token_walk(phi, second, 0, '[{"i":1,"g":0,"a":"N","e":[', prompt=first)
    assert not set(base.eos_token_ids).intersection(partial)
    complete, allowed = token_walk(phi, second, 0, '[{"i":1,"g":0,"a":"N","e":[1]}]', prompt=first)
    assert set(base.eos_token_ids) <= set(allowed)
    for eos in base.eos_token_ids: assert second(0, torch.tensor(complete + [eos]))
    third = wrapped.for_tasks([({"refs": [{}]}, [{}])])
    _, allowed = token_walk(phi, third, 0, empty, prompt=complete)
    assert set(base.eos_token_ids) <= set(allowed)


def test_wrapper_rejects_empty_batch_unknown_rows_and_inconsistent_sources(phi):
    import torch
    wrapped = phi[2]
    with pytest.raises(ValueError): wrapped.for_tasks([])
    with pytest.raises(ValueError): wrapped.for_tasks([({"refs": [], "body": {"excerpts": ["hidden"]}}, [{}])])
    callback = wrapped.for_tasks([({"refs": [{}]}, [{}])])
    for row in (-1, 1, True):
        with pytest.raises(ValueError): callback(row, torch.tensor([1]))


def test_protocol_discloses_structural_constraints_and_dependency():
    protocol = decoding.decoding_protocol()
    assert protocol["field_order"] == ["i", "g", "a", "e"]
    assert protocol["nonempty_evidence"]["combinations"] == 15
    assert protocol["empty_evidence_abstention_is_enforced_not_measured_model_ability"]
    assert protocol["known_answer_citation_presence_is_enforced_not_measured_entailment"]
    assert protocol["prospective_control_and_source_validation_required_before_activation"]
    assert not any(protocol[key] for key in ("semantic_entailment_verified", "citations_inserted", "scores_used", "posthoc_answer_fixes"))
    assert len(protocol["implementation_sha256"]) == len(protocol["schema_template_sha256"]) == 64
    assert protocol["dependency_implementation_sha256"]["frontier_evidence_first_decoding.py"] == hashlib.sha256((ROOT / "scripts/frontier_evidence_first_decoding.py").read_bytes()).hexdigest()
