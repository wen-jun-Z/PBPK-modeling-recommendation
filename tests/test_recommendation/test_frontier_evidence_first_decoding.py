"""CPU character and real official Phi tokenizer tests of evidence-first JSON."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import frontier_evidence_first_decoding as decoding
import frontier_judge_decoding as original
import run_frontier_assessment as assessment


def advance(parser, text):
    for char in text:
        if char not in parser.get_allowed_characters():
            raise ValueError((text, char, parser.get_allowed_characters()))
        parser = parser.add_character(char)
    return parser


def accepts(text, refs=3, requests=1):
    try:
        return advance(decoding.evidence_first_parser(requests, refs), text).can_end()
    except ValueError:
        return False


@pytest.mark.parametrize("grade,applicability", itertools.product([None, 0, 1, 2, 3], ["Y", "N", "U"]))
def test_every_original_valid_answer_object_has_evidence_first_serialization(grade, applicability):
    for size in range(1, 4):
        for refs in itertools.permutations([1, 2, 3], size):
            old = [{"i": 1, "g": grade, "a": applicability, "e": list(refs)}]
            new = [{key: old[0][key] for key in decoding.FIELD_ORDER}]
            assert accepts(json.dumps(new))
            assert assessment.parse_answers(json.dumps(new), [1], [{}] * 3) == assessment.parse_answers(json.dumps(old), [1], [{}] * 3)


@pytest.mark.parametrize("refs", [0, 1, 3, 32])
def test_empty_evidence_always_allows_unknown_without_inserting_citations(refs):
    assert accepts('[{"i":1,"e":[],"g":null,"a":"U"}]', refs=refs)
    for grade, app in itertools.product([None, 0, 1, 2, 3], ["Y", "N", "U"]):
        answer = json.dumps([dict(i=1, e=[], g=grade, a=app)])
        assert accepts(answer, refs=refs) is (grade is None and app == "U")


@pytest.mark.parametrize("raw,refs", [
    ('[{"i":1,"e":[1],"g":null,"a":"U"}]', 0),
    ('[{"i":1,"e":[0],"g":0,"a":"N"}]', 3),
    ('[{"i":1,"e":[4],"g":0,"a":"N"}]', 3),
    ('[{"i":1,"e":[1,1],"g":0,"a":"N"}]', 12),
    ('[{"i":1,"e":[12,12],"g":0,"a":"N"}]', 12),
    ('[{"i":1,"e":[1,2,3,4],"g":0,"a":"N"}]', 32),
    ('[{"i":1,"e":[1,],"g":0,"a":"N"}]', 3),
    ('[{"i":1,"e":[1.0],"g":0,"a":"N"}]', 3),
    ('[{"i":1,"e":[1e0],"g":0,"a":"N"}]', 3),
    ('[{"i":1,"e":[true],"g":0,"a":"N"}]', 3),
    ('[{"i":1,"e":["1"],"g":0,"a":"N"}]', 3),
    ('[{"i":1,"g":0,"a":"N","e":[1]}]', 3),
    ('[{"e":[1],"i":1,"g":0,"a":"N"}]', 3),
    ('[{"i":1,"e":[1],"a":"N","g":0}]', 3),
    ('[{"i":1,"e":[1],"g":4,"a":"N"}]', 3),
    ('[{"i":1,"e":[1],"g":0,"a":"Maybe"}]', 3),
    ('[{"i":1,"e":[1],"g":0,"a":"N","reason":"x"}]', 3),
    ('[{"i":1,"e":[],"g":null,"a":"U"}] explanation', 3),
    ('```json\n[{"i":1,"e":[],"g":null,"a":"U"}]\n```', 3),
    ('[{"i":1,"\\u0065":[],"g":3,"a":"Y"}]', 3),
    ('[]', 3),
])
def test_invalid_reference_schema_order_and_escaped_key_bypass_rejected(raw, refs):
    assert not accepts(raw, refs=refs)


def test_multidigit_uniqueness_keeps_other_ids_sharing_the_same_prefix():
    p = advance(decoding.evidence_first_parser(1, 12), '[{"i":1,"e":[1,1')
    allowed = p.get_allowed_characters()
    assert "]" not in allowed and "," not in allowed
    assert {"0", "1", "2"}.issubset(set(allowed))
    for refs in ([1, 10], [1, 11], [1, 12], [12, 1], [12, 10, 11]):
        assert accepts(json.dumps([dict(i=1, e=refs, g=3, a="Y")]), refs=12)


def test_request_array_retains_original_index_order_freedom_and_count():
    rows = [dict(i=2, e=[], g=None, a="U"), dict(i=1, e=[2], g=0, a="N")]
    raw = json.dumps(rows)
    assert accepts(raw, refs=2, requests=2)
    assert set(assessment.parse_answers(raw, [1, 2], [{}, {}])) == {1, 2}
    assert not accepts(raw, refs=2, requests=1)
    assert not accepts(json.dumps(rows[:1]), refs=2, requests=2)
    rows.append(dict(i=3, e=[1, 2], g=None, a="Y"))
    assert accepts(json.dumps(rows), refs=2, requests=3)


def test_parser_states_are_immutable_and_whitespace_is_bounded():
    start = decoding.evidence_first_parser(1, 3)
    branch = advance(start, '[{"i":1,"e":[]')
    assert "[" in start.get_allowed_characters() and not start.can_end()
    assert not branch.can_end()
    assert accepts('[\n { "i" : 1, "e" : [ 1, 2 ], "g" : 3, "a" : "Y" }]')
    assert not accepts('[   {"i":1,"e":[],"g":null,"a":"U"}]')


@pytest.mark.parametrize("counts", [(0, 1), (True, 1), (1, True), (1, -1), (1, 1.0)])
def test_invalid_counts_rejected(counts):
    with pytest.raises(ValueError):
        decoding.evidence_first_parser(*counts)


@pytest.fixture(scope="module")
def phi():
    """Use actual pinned project tokenizer and EOS configuration, never weights/GPU."""
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
    return tokenizer, base, decoding.evidence_first_constraint(base)


def token_walk(phi, callback, batch_id, output, prompt=None):
    import torch
    tokenizer, _, _ = phi
    ids = list(prompt if prompt is not None else tokenizer.encode("A private prompt, not JSON.", add_special_tokens=False))
    for token in tokenizer.encode(output, add_special_tokens=False):
        allowed = callback(batch_id, torch.tensor(ids, dtype=torch.long))
        assert token in allowed, (output, tokenizer.decode([token]), allowed)
        ids.append(token)
    return ids, callback(batch_id, torch.tensor(ids, dtype=torch.long))


@pytest.mark.parametrize("grade,app", itertools.product([None, 0, 1, 2, 3], ["Y", "N", "U"]))
def test_official_phi_tokens_preserve_all_fifteen_nonempty_decisions(phi, grade, app):
    _, base, wrapped = phi
    callback = wrapped.for_tasks([({"refs": [{}] * 12}, [{}])])
    _, allowed = token_walk(phi, callback, 0, json.dumps([dict(i=1, e=[1, 12, 10], g=grade, a=app)], separators=(",", ":")))
    assert set(base.eos_token_ids).issubset(allowed)


def test_official_phi_empty_source_and_identical_prompt_rows_are_isolated(phi):
    import torch
    _, base, wrapped = phi
    assert wrapped.tokenizer_data is base.tokenizer_data
    assert wrapped.eos_token_ids is base.eos_token_ids
    callback = wrapped.for_tasks([({"refs": [], "body": {"excerpts": []}}, [{}]),
                                  ({"refs": [{}] * 12}, [{}])])
    empty = '[{"i":1,"e":[],"g":null,"a":"U"}]'
    cited = '[{"i":1,"e":[12],"g":3,"a":"Y"}]'
    _, allowed_empty = token_walk(phi, callback, 0, empty)
    _, allowed_cited = token_walk(phi, callback, 1, cited)
    assert set(base.eos_token_ids).issubset(allowed_empty)
    assert set(base.eos_token_ids).issubset(allowed_cited)
    seq, _ = token_walk(phi, callback, 0, '[{"i":1,"e":[')
    next_tokens = callback(0, torch.tensor(seq))
    assert not any(phi[0].decode([t]).lstrip().startswith("1") for t in next_tokens)


@pytest.mark.parametrize("raw", [
    '[{"i":1,"e":[1,1],"g":0,"a":"N"}]',
    '[{"i":1,"e":[1,12,1],"g":0,"a":"N"}]',
    '[{"i":1,"e":[13],"g":0,"a":"N"}]',
    '[{"i":1,"e":[],"g":3,"a":"Y"}]',
    '[{"i":1,"g":0,"a":"N","e":[1]}]',
])
def test_official_phi_tokens_cannot_bypass_reference_or_empty_evidence_constraints(phi, raw):
    callback = phi[2].for_tasks([({"refs": [{}] * 12}, [{}])])
    with pytest.raises(AssertionError):
        token_walk(phi, callback, 0, raw)


def test_official_phi_each_call_fresh_and_eos_forbidden_before_complete(phi):
    import torch
    _, base, wrapped = phi
    tasks = [({"refs": [{}]}, [{}])]
    first = wrapped.for_tasks(tasks)
    partial, allowed = token_walk(phi, first, 0, '[{"i":1,"e":[],"g":null')
    assert not set(base.eos_token_ids).intersection(allowed)
    complete, allowed = token_walk(phi, first, 0, '[{"i":1,"e":[],"g":null,"a":"U"}]')
    assert set(base.eos_token_ids).issubset(allowed)
    for eos in base.eos_token_ids:
        assert first(0, torch.tensor(complete + [eos]))
    second = wrapped.for_tasks(tasks)
    # Reuse a prior complete output as a new prompt; parser state must start fresh.
    next_tokens = second(0, torch.tensor(complete))
    assert not set(base.eos_token_ids).intersection(next_tokens)
    _, done = token_walk(phi, second, 0, '[{"i":1,"e":[1],"g":0,"a":"N"}]', prompt=complete)
    assert set(base.eos_token_ids).issubset(done)


def test_wrapper_rejects_empty_batch_unknown_rows_and_inconsistent_empty_source(phi):
    import torch
    wrapped = phi[2]
    with pytest.raises(ValueError):
        wrapped.for_tasks([])
    with pytest.raises(ValueError):
        wrapped.for_tasks([({"refs": [], "body": {"excerpts": ["hidden"]}}, [{}])])
    callback = wrapped.for_tasks([({"refs": [{}]}, [{}])])
    for index in (-1, 1, True):
        with pytest.raises(ValueError):
            callback(index, torch.tensor([1]))


def test_protocol_discloses_forced_abstention_and_does_not_claim_semantic_validation():
    p = decoding.decoding_protocol()
    assert p["field_order"] == ["i", "e", "g", "a"]
    assert p["nonempty_evidence"]["combinations"] == 15
    assert p["empty_evidence_abstention_is_enforced_not_measured_model_ability"]
    assert not p["semantic_entailment_verified"]
    assert not p["citations_inserted"] and not p["scores_used"] and not p["posthoc_answer_fixes"]
    assert len(p["implementation_sha256"]) == len(p["schema_template_sha256"]) == 64
