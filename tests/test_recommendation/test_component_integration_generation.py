"""Input isolation and generation contracts; no GPU/model inference."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_component_model_integration as prepare
import generate_component_model_integration as generate
from pbpk_kb.recommendation.online_agent import _prompt_row
from pbpk_kb.recommendation.schemas import ModelingRequest


def query():
    return {"query_id": "clinical:unit", "compounds": ["caffeine"],
            "task_type": "plasma_concentration_time_prediction", "text": "Predict caffeine at 50 mg, fasted.",
            "features": {"species": [], "routes": ["oral"], "endpoints": ["cmax"], "enzymes": ["cyp1a2"]}}


def test_request_preserves_missing_species_and_true_condition_differences():
    q = query()
    a = prepare.make_request(q, {"normalized_design": {"dose": {"value": 50, "unit": "mg"}, "food": "fasted"}})
    b = prepare.make_request(q, {"normalized_design": {"dose": {"value": 250, "unit": "mg"}, "food": "fed"}})
    assert a["species"] == "unspecified"
    assert a["dose"] == 50 and b["dose"] == 250
    assert a["fed_or_fasted"] == "fasted" and b["fed_or_fasted"] == "fed"
    assert prepare.digest(a) != prepare.digest(b)


def test_observations_and_evaluation_errors_do_not_enter_request():
    original = {"normalized_design": {"dose": {"value": 50, "unit": "mg"}}}
    poisoned = deepcopy(original)
    poisoned.update(observations=["FORBIDDEN_OBSERVATION_192"], model_scores=["FORBIDDEN_SCORE_384"])
    poisoned["normalized_design"].update(evaluation_target={"observed_Cmax": "FORBIDDEN_666"},
                                          prediction_contract={"best_model": "FORBIDDEN_CANDIDATE"})
    assert prepare.make_request(query(), original) == prepare.make_request(query(), poisoned)


def test_multiple_drugs_are_not_silently_collapsed():
    q = query()
    q["compounds"].append("rifampicin")
    with pytest.raises(ValueError, match="unique target"):
        prepare.make_request(q)


def test_registered_denominators_no_silent_drops():
    entries = prepare.rows(prepare.OUT / "inputs/requests.jsonl")
    assert len(entries) == len({e["query_id"] for e in entries}) == 186
    assert sum(e["cohort"] == "core" for e in entries) == 145
    assert sum(e["cohort"] == "execution" for e in entries) == 41
    for entry in entries:
        assert entry["request_sha256"] == prepare.digest(entry["modeling_request"])
        assert entry["raw_query_sha256"] == prepare.digest(entry["query_record"])
        if entry["status"] == "unavailable":
            assert entry["reason"]


def test_target_publication_and_public_models_absent_from_filtered_graph():
    mask = prepare.read(prepare.OUT / "inputs/graph_mask.json")
    excluded = {eid for eid, reason in mask["rejected_evidence"].items() if reason == "target_publication_trial_or_raw_batch"}
    assert len(excluded) >= 2
    corpus = generate.corpus_for(prepare.OUT)
    assert not any(n["node_type"] == "PBPKModel" for n in corpus.nodes)
    assert not (excluded & set(corpus.edges_by_evidence))
    assert all((e.get("provenance") or {}).get("evidence_id") in mask["allowed_evidence_ids"] for e in corpus.edges)


def test_real_prompt_uses_text_only_and_exact_request_binding():
    corpus = generate.corpus_for(prepare.OUT)
    entry = next(e for e in prepare.rows(prepare.OUT / "inputs/requests.jsonl")
                 if e["status"] == "pending_retrieval" and e["query_record"]["compounds"] == ["caffeine"])
    request = ModelingRequest.model_validate(entry["modeling_request"])
    # CPU fixture for prompt assembly only: production rankings use HGT.
    retrieved = corpus.retrieve(request)
    ranking = {"selected_subgraph_ids": [s.subgraph_id for s in retrieved], "ranked_candidates": [
        {"subgraph_id": s.subgraph_id, "final_score": s.score, "rrf_score": .1, "reranker_score": .5,
         "teacher_score": .2, "raw_hgt_score": .3, "residual_alpha": .1, "graph_constraint_pass": True} for s in retrieved]}
    row, _ = generate.online_prompt(corpus, request, ranking, entry, _prompt_row)
    assert set(row["prompts"]) == {"text_graph"}
    assert "SOFT_GRAPH_TOKEN_CATALOG" not in row["prompts"]["text_graph"]
    assert row["modeling_request"] == entry["modeling_request"]
    assert row["request_sha256"] == entry["request_sha256"]
    assert row["raw_query_sha256"] == entry["raw_query_sha256"]
    assert row["allowed_subgraph_ids"] == row["retrieved_subgraph_ids"]
    assert row["allowed_output_ids"]["public_model_ids"] == []
    assert generate.selection_schema(row)["properties"]["public_model_ids"]["maxItems"] == 0


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [self(t, add_special_tokens=add_special_tokens)["input_ids"] for t in text]}
        result = {"input_ids": list(range(len(text) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def test_dense_chunking_preserves_all_characters_and_instruction_budget():
    tokenizer = CharacterTokenizer()
    text = "\u539f\u59cb\u5242\u91cf50mg/kg；\u98df\u7269fed；" * 12 + "END"
    pieces = generate.complete_text_chunks(tokenizer, text, prefix="QUERY: ", limit=32)
    assert len(pieces) > 1
    assert "".join(p for p, _ in pieces) == text
    assert all(generate.token_count(tokenizer, "QUERY: " + p) <= 32 for p, _ in pieces)
    assert sum(weight for _, weight in pieces) == len(text)


def test_dense_single_short_text_remains_unmodified():
    text = "caffeine 50 mg"
    assert generate.complete_text_chunks(CharacterTokenizer(), text, prefix="Q: ", limit=64) == [(text, len(text))]


def test_reranker_context_is_measured_and_never_forced_under_model_limit():
    assert generate.select_context_limit([2400, 3900], 32768) == 4096
    assert generate.select_context_limit([4200], 32768) == 8192
    with pytest.raises(ValueError, match="No truncation"):
        generate.select_context_limit([33000], 32768)


def test_complete_reranker_lengths_include_prompt_and_suffix():
    class Fake:
        tokenizer = CharacterTokenizer()
        prefix_tokens = [1] * 10
        suffix_tokens = [2] * 5
        _format = staticmethod(lambda q, d: q + d)
    assert generate.complete_pair_lengths(Fake(), [("abc", "de"), ("x", "y")]) == [20, 17]


def test_reranker_oom_halving_preserves_pair_order_and_resume():
    from types import SimpleNamespace
    class OOM(RuntimeError):
        pass
    class Fake:
        torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None))
        calls = []
        def score(self, pairs):
            self.calls.append(len(pairs))
            if len(pairs) > 2:
                raise OOM("bounded test")
            return [float(q) / 10 for q, _ in pairs], [
                {"original_token_count": len(d), "model_input_token_count": len(d), "truncated": False}
                for _, d in pairs]
    pairs = [(str(i), "d" * (10 - i)) for i in range(6)]
    checkpoint = []
    fake = Fake()
    values, audit = generate.score_complete_pairs(fake, pairs, [len(d) for _, d in pairs],
        maximum_batch=8, checkpoint=lambda done: checkpoint.append(deepcopy(done)))
    assert fake.calls[:3] == [6, 3, 1]
    assert values == [i / 10 for i in range(6)]
    assert len(audit) == 6 and len(checkpoint[-1]) == 6
    resumed = Fake()
    values2, _ = generate.score_complete_pairs(resumed, pairs, [len(d) for _, d in pairs], initial_records=checkpoint[-1])
    assert values2 == values


def test_reranker_audit_rejects_any_truncation_even_after_preflight():
    from types import SimpleNamespace
    class Fake:
        torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=MemoryError))
        def score(self, pairs):
            return [.5], [{"original_token_count": 100, "model_input_token_count": 64, "truncated": True}]
    with pytest.raises(ValueError, match="truncated or changed"):
        generate.score_complete_pairs(Fake(), [("q", "d")], [100])
