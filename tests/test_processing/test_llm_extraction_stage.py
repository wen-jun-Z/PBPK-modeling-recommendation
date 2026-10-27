from __future__ import annotations

import json
import pytest
from types import SimpleNamespace

from pbpk_kb.processing.llm_agent import LLMExtractionAgent
from pbpk_kb.processing.run_llm_extraction import (
    CanonicalSentence,
    PendingTask,
    UpstreamIndex,
    _canonical_sentence_rank,
    _publication_duplicate_reason,
    _resolve_ineligible_api_failures,
    _resolve_target_compound,
    _validate_candidates,
)


def _chunk() -> dict[str, object]:
    return {
        "run_id": "run-test",
        "source_id": "pmc_oa",
        "compound": "midazolam",
        "raw_asset_id": "asset-1",
        "parsed_document_id": "doc-1",
        "parsed_table_id": None,
        "chunk_id": "chunk-1",
        "storage_path": "/raw/article.xml",
        "source_url": "https://example.test/article",
        "checksum": "abc123",
        "source_location": "pmc_oa:article.xml#section=results#chunk=1",
        "chunk_text": "After a 10 mg oral dose, midazolam reached a mean Cmax of 53 ng/mL.",
    }


def _upstream() -> UpstreamIndex:
    return UpstreamIndex(
        raw_assets={
            "asset-1": {
                "raw_asset_id": "asset-1",
                "checksum": "abc123",
                "storage_path": "/raw/article.xml",
            }
        },
        parsed_document_ids={"doc-1"},
        parsed_table_ids=set(),
        target_compounds={"midazolam", "warfarin"},
    )


def _task() -> PendingTask:
    return PendingTask(
        chunk=_chunk(),
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-1", "attempt_number": 1},
        input_hash="hash-1",
    )


@pytest.mark.parametrize("quote,low,high,unit,allowed", [
    ("Caffeine half-life increases from an average of 3 h for non-pregnant women to 10.5 h during pregnancy.", 3, 10.5, "h", False),
    ("Normal (488 +/- 63 ml/min) or impaired (338 +/- 30 ml/min) caffeine clearance.", 338, 488, "ml/min", False),
    ("Caffeine half-life increased from 3 h to 10.5 h after treatment.", 3, 10.5, "h", False),
    ("Caffeine half-life was 3 h and 10.5 h in the two groups.", 3, 10.5, "h", False),
    ("Caffeine half-life was 140-270 minutes.", 140, 270, "minutes", True),
    ("Caffeine half-life increased to 160-240 minutes.", 160, 240, "minutes", True),
    ("Caffeine half-life ranged from 30 minutes to 2 hours.", .5, 2, "hours", True),
    ("Caffeine half-life ranged between 3 and 10.5 h.", 3, 10.5, "h", True),
    ("Caffeine AUC had a confidence interval of 87.9%-102%.", 87.9, 102, "%", True),
])
def test_group_values_are_never_promoted_to_intervals(quote, low, high, unit, allowed):
    from pbpk_kb.processing.run_llm_extraction import _numeric_range_source_error
    row = SimpleNamespace(value=None, value_low=low, value_high=high, unit=unit,
                          parameter="half_life" if "half-life" in quote else "AUC", source_text=quote)
    assert (_numeric_range_source_error(row) is None) is allowed


def test_effect_on_other_substance_is_not_target_adme():
    from pbpk_kb.processing.run_llm_extraction import _parameter_owner_error
    assert _parameter_owner_error("adme", "excretion", "Caffeine increased urinary epinephrine excretion.", "caffeine")
    assert _parameter_owner_error("adme", "excretion", "Caffeine increased its own excretion.", "caffeine") is None
    assert _parameter_owner_error("adme", "excretion", "Urinary caffeine excretion was increased.", "caffeine") is None
    assert _parameter_owner_error("ddi", "excretion", "Caffeine increased urinary epinephrine excretion.", "caffeine") is None


@pytest.mark.parametrize('therapy', ['Continuous kidney replacement therapy', 'CRRT', 'Hemoadsorption', 'Renal replacement therapy'])
def test_extracorporeal_clearance_is_not_intrinsic_drug_clearance(therapy):
    from pbpk_kb.processing.run_llm_extraction import _clearance_context_error
    assert _clearance_context_error('clearance', therapy + ' provided lower caffeine clearance.', 'caffeine')
    assert _clearance_context_error('clearance', 'Caffeine clearance was lower in premature infants.', 'caffeine') is None


def test_quote_cut_inside_unit_is_rejected_even_when_in_old_input_items():
    task = _task()
    task.chunk['chunk_text'] = 'Midazolam clearance and volume of distribution were 0.0476 L. h -1 and 1.1 L.'
    quote = task.chunk['chunk_text'].split(' h -1')[0]
    task.request_record['input_items'] = [quote]
    candidate = {'target_compound': 'midazolam', 'evidence_type': 'clinical_pk',
                 'parameter': 'volume_of_distribution', 'source_text': quote,
                 'value': None, 'unit': None, 'condition': None, 'confidence': .7, 'requires_review': True}
    valid, rejected = _validate_candidates(task, [candidate], _upstream())
    assert valid == []
    assert 'source_quote_truncated_inside_compound_unit' in rejected[0]['errors']


def test_routing_cache_reuses_immutable_inputs_and_invalidates_changes(tmp_path, monkeypatch):
    from collections import Counter
    from pbpk_kb.processing import run_llm_extraction as module
    names = ['raw_inventory.jsonl', 'parsed_documents.jsonl', 'parsed_tables.jsonl', 'text_chunks.jsonl']
    for name in names:
        (tmp_path/name).write_text('{}\n')
    layout = SimpleNamespace(path=lambda name: tmp_path/name)
    calls = []
    def build(*args, **kwargs):
        calls.append(1)
        return {}, Counter(canonical_sentences=3)
    monkeypatch.setattr(module, '_CANONICAL_INDEX_CACHE', {})
    monkeypatch.setattr(module, '_build_canonical_sentence_index', build)
    first = module._cached_canonical_sentence_index(layout, min_chars=160, upstream=_upstream())
    first[1]['canonical_sentences'] = 999
    assert module._cached_canonical_sentence_index(layout, min_chars=160, upstream=_upstream())[1]['canonical_sentences'] == 3
    assert len(calls) == 1
    (tmp_path/'parsed_documents.jsonl').write_text('{"changed": true}\n')
    module._cached_canonical_sentence_index(layout, min_chars=160, upstream=_upstream())
    module._cached_canonical_sentence_index(layout, min_chars=40, upstream=_upstream())
    monkeypatch.setattr(module, 'LLM_ROUTING_VERSION', 'changed-routing')
    module._cached_canonical_sentence_index(layout, min_chars=40, upstream=_upstream())
    assert len(calls) == 4


def test_input_hash_is_scoped_to_target_compound() -> None:
    agent = LLMExtractionAgent(mode="off")
    first = agent.build_request_record({"target_compound": "midazolam", "input_text_or_table": "same text"})
    second = agent.build_request_record({"target_compound": "warfarin", "input_text_or_table": "same text"})

    assert first["input_hash"] != second["input_hash"]
    assert agent.prompt_template_version == "pbpk-evidence-agent-v14-local-qwen"


def test_pubmed_abstract_is_skipped_when_matching_pmc_fulltext_exists() -> None:
    upstream = _upstream()
    upstream.document_metadata["doc-1"] = {"pmcid": "PMC123", "source_id": "pubmed", "content_kind": "abstract"}
    upstream.pmc_fulltext_ids.add("PMC123")
    chunk = _chunk()
    chunk["source_id"] = "pubmed"

    assert _publication_duplicate_reason(chunk, upstream) == "skipped_pubmed_abstract_with_pmc_fulltext"


def test_canonical_sentence_prefers_dailymed_over_openfda() -> None:
    dailymed = CanonicalSentence("daily", "dailymed", "dailymed:label#chunk=1")
    openfda = CanonicalSentence("fda", "openfda_drug_label", "openfda:label#chunk=1")

    assert _canonical_sentence_rank(dailymed) < _canonical_sentence_rank(openfda)


def test_valid_candidate_requires_target_quote_number_and_canonical_unit() -> None:
    candidates = [
        {
            "target_compound": "midazolam",
            "evidence_type": "clinical_pk",
            "parameter": "Cmax",
            "value": 53,
            "unit": "ng/mL",
            "condition": "10 mg oral dose",
            "source_text": "After a 10 mg oral dose, midazolam reached a mean Cmax of 53 ng/mL.",
            "confidence": 0.91,
            "requires_review": True,
        }
    ]

    valid, rejected = _validate_candidates(_task(), candidates, _upstream())

    assert len(valid) == 1
    assert rejected == []
    assert valid[0]["target_compound"] == "midazolam"


def test_wrong_target_and_unmapped_unit_are_rejected() -> None:
    candidates = [
        {
            "target_compound": "warfarin",
            "evidence_type": "clinical_pk",
            "parameter": "Cmax",
            "value": 53,
            "unit": "widgets",
            "condition": None,
            "source_text": "After a 10 mg oral dose, midazolam reached a mean Cmax of 53 ng/mL.",
            "confidence": 0.5,
            "requires_review": True,
        }
    ]

    valid, rejected = _validate_candidates(_task(), candidates, _upstream())

    assert valid == []
    errors = set(rejected[0]["errors"])
    assert "candidate_target_compound_does_not_match_request" in errors
    assert "unit_not_in_canonical_unit_map" in errors
    assert "unit_not_copied_from_source_text" in errors


def test_numeric_hallucination_is_rejected() -> None:
    candidates = [
        {
            "target_compound": "midazolam",
            "evidence_type": "clinical_pk",
            "parameter": "Cmax",
            "value": 99,
            "unit": "ng/mL",
            "condition": None,
            "source_text": "After a 10 mg oral dose, midazolam reached a mean Cmax of 53 ng/mL.",
            "confidence": 0.5,
            "requires_review": True,
        }
    ]

    valid, rejected = _validate_candidates(_task(), candidates, _upstream())

    assert valid == []
    assert "numeric_value_not_found_in_source_text" in rejected[0]["errors"]


def test_numeric_range_and_pk_unit_are_validated() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "Midazolam clearance ranged from 0.02 L/h to 0.10 L/h in adults."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-range", "attempt_number": 1},
        input_hash="hash-range",
    )
    candidates = [
        {
            "target_compound": "midazolam",
            "evidence_type": "clinical_pk",
            "parameter": "clearance",
            "value": None,
            "value_low": 0.02,
            "value_high": 0.10,
            "unit": "L/h",
            "condition": "adults",
            "source_text": chunk["chunk_text"],
            "confidence": 0.8,
            "requires_review": True,
        }
    ]

    valid, rejected = _validate_candidates(task, candidates, _upstream())

    assert len(valid) == 1
    assert rejected == []


def test_incomplete_source_assertion_is_rejected() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "The half-life of midazolam increased from"
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-incomplete", "attempt_number": 1},
        input_hash="hash-incomplete",
    )
    candidates = [
        {
            "target_compound": "midazolam",
            "evidence_type": "ddi",
            "parameter": "half-life",
            "value": None,
            "unit": None,
            "condition": "diltiazem",
            "source_text": chunk["chunk_text"],
            "confidence": 0.5,
            "requires_review": True,
        }
    ]

    valid, rejected = _validate_candidates(task, candidates, _upstream())

    assert valid == []
    assert "source_text_is_incomplete_assertion" in rejected[0]["errors"]


def test_unfinished_target_condition_is_rejected_after_llm() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "The half-life of midazolam increased from 5 to 7 hours when midazolam was taken"
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-unfinished-condition", "attempt_number": 1},
        input_hash="hash-unfinished-condition",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "ddi",
        "parameter": "half_life",
        "value": None,
        "unit": None,
        "condition": "coadministration",
        "source_text": chunk["chunk_text"],
        "confidence": 0.5,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert valid == []
    assert "source_text_is_incomplete_assertion" in rejected[0]["errors"]


def test_pharmacodynamic_endpoint_is_not_clinical_pk() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "Midazolam depressed the ventilatory response to carbon dioxide for 15 minutes."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-pd", "attempt_number": 1},
        input_hash="hash-pd",
    )
    candidates = [
        {
            "target_compound": "midazolam",
            "evidence_type": "clinical_pk",
            "parameter": "ventilatory_response_to_carbon_dioxide",
            "value": None,
            "unit": None,
            "condition": "adults",
            "source_text": chunk["chunk_text"],
            "confidence": 0.8,
            "requires_review": True,
        }
    ]

    valid, rejected = _validate_candidates(task, candidates, _upstream())

    assert valid == []
    assert "parameter_outside_evidence_type_domain" in rejected[0]["errors"]


def test_special_population_safety_or_dosing_label_is_not_pk_evidence() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "Midazolam may cause neonatal sedation when used late in pregnancy."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-safety", "attempt_number": 1},
        input_hash="hash-safety",
    )
    candidates = [
        {
            "target_compound": "midazolam",
            "evidence_type": "special_population",
            "parameter": "pregnancy_safety",
            "value": None,
            "unit": None,
            "condition": "pregnancy",
            "source_text": chunk["chunk_text"],
            "confidence": 0.8,
            "requires_review": True,
        }
    ]

    valid, rejected = _validate_candidates(task, candidates, _upstream())

    assert valid == []
    assert "parameter_outside_evidence_type_domain" in rejected[0]["errors"]


def test_cmax_label_requires_cmax_or_peak_terminology_in_source() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "Cimetidine increased the mean midazolam steady-state concentration from 57 to 71 ng/mL."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-cmax-mismatch", "attempt_number": 1},
        input_hash="hash-cmax-mismatch",
    )
    candidates = [
        {
            "target_compound": "midazolam",
            "evidence_type": "ddi",
            "parameter": "Cmax",
            "value": 71,
            "unit": "ng/mL",
            "condition": "cimetidine",
            "source_text": chunk["chunk_text"],
            "confidence": 0.8,
            "requires_review": True,
        }
    ]

    valid, rejected = _validate_candidates(task, candidates, _upstream())

    assert valid == []
    assert "parameter_not_supported_by_source_text_terminology" in rejected[0]["errors"]


def test_direct_percent_change_is_not_manufactured_when_llm_omits_value() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "Saquinavir caused a 56% reduction in the clearance of midazolam."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-percent", "attempt_number": 1},
        input_hash="hash-percent",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "ddi",
        "parameter": "clearance",
        "value": None,
        "unit": None,
        "condition": "saquinavir",
        "source_text": chunk["chunk_text"],
        "confidence": 0.8,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())
    assert not valid
    assert "direct_percent_change_missing_value" in rejected[0]["errors"]
    assert rejected[0]["candidate"]["value"] is None

    candidate.update({"value": 56, "unit": "% reduction"})
    valid, rejected = _validate_candidates(task, [candidate], _upstream())
    assert len(valid) == 1
    assert rejected == []


def test_bounded_percent_only_repairs_qualifier_of_an_existing_value() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "The amount of midazolam excreted unchanged in urine was less than 0.5%."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-bounded-percent", "attempt_number": 1},
        input_hash="hash-bounded-percent",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "adme",
        "parameter": "excretion",
        "value": None,
        "unit": None,
        "condition": None,
        "source_text": chunk["chunk_text"],
        "confidence": 0.8,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert rejected == []
    assert valid[0]["value"] is None and valid[0]["unit"] is None
    candidate.update(value=0.5, unit="%")
    valid, rejected = _validate_candidates(task, [candidate], _upstream())
    assert not rejected and valid[0]["value"] == 0.5
    assert valid[0]["value_qualifier"] == "<"
    assert valid[0]["validation_repairs"] == ["bounded_percent_qualifier_from_source_text"]


def test_alcoholic_population_fold_only_repairs_unit_of_an_existing_value() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "The mean half-life of midazolam increased 2.5-fold in alcoholic patients."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-alcohol-fold", "attempt_number": 1},
        input_hash="hash-alcohol-fold",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "special_population",
        "parameter": "half_life",
        "value": None,
        "unit": None,
        "condition": "alcoholic patients",
        "source_text": chunk["chunk_text"],
        "confidence": 0.85,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert rejected == []
    assert valid[0]["value"] is None and valid[0]["unit"] is None
    candidate.update(value=2.5)
    valid, rejected = _validate_candidates(task, [candidate], _upstream())
    assert not rejected and valid[0]["value"] == 2.5
    assert valid[0]["unit"] == "fold"
    assert valid[0]["validation_repairs"] == ["direct_fold_unit_from_source_text"]


def test_result_only_quote_expands_to_unique_adjacent_target_context() -> None:
    context = "The pharmacokinetics of midazolam were evaluated in healthy adults."
    result = "The volume of distribution ranged from 1.0 to 3.1 L/kg."
    chunk = _chunk()
    chunk["chunk_text"] = f"{context} {result}"
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={
            "request_id": "request-context-expansion",
            "attempt_number": 1,
            "input_items": [chunk["chunk_text"]],
        },
        input_hash="hash-context-expansion",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "clinical_pk",
        "parameter": "volume_of_distribution",
        "value": None,
        "value_low": 1.0,
        "value_high": 3.1,
        "unit": "L/kg",
        "condition": "healthy adults",
        "source_text": result,
        "confidence": 0.8,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert rejected == []
    assert valid[0]["source_text"] == chunk["chunk_text"]
    assert valid[0]["validation_repairs"] == ["expanded_source_text_to_complete_input_item"]


def test_target_named_truncated_quote_expands_to_complete_input_item() -> None:
    complete_item = (
        "The mean bioavailability of theophylline during a single oral dose was 97%, "
        "which did not significantly differ from the i.v. reference."
    )
    chunk = _chunk()
    chunk.update(
        {
            "compound": "theophylline",
            "chunk_text": complete_item,
        }
    )
    upstream = _upstream()
    upstream.target_compounds.add("theophylline")
    task = PendingTask(
        chunk=chunk,
        target_compound="theophylline",
        status_record={},
        request_record={
            "request_id": "request-complete-item-expansion",
            "attempt_number": 1,
            "input_items": [complete_item],
        },
        input_hash="hash-complete-item-expansion",
    )
    candidate = {
        "target_compound": "theophylline",
        "evidence_type": "adme",
        "parameter": "bioavailability",
        "value": 97.0,
        "unit": "%",
        "condition": "single oral dose",
        "source_text": complete_item.removesuffix(" reference."),
        "confidence": 0.8,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], upstream)

    assert rejected == []
    assert valid[0]["source_text"] == complete_item
    assert valid[0]["validation_repairs"] == ["expanded_source_text_to_complete_input_item"]


def test_joint_parent_and_metabolite_pk_subject_is_rejected() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = (
        "Cmax for midazolam and its 1-hydroxy metabolite was approximately one-half of the "
        "intravenous value."
    )
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-joint-metabolite", "attempt_number": 1},
        input_hash="hash-joint-metabolite",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "clinical_pk",
        "parameter": "Cmax",
        "value": None,
        "unit": None,
        "condition": "intramuscular administration",
        "source_text": chunk["chunk_text"],
        "confidence": 0.7,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert valid == []
    assert "joint_parent_and_metabolite_subject" in rejected[0]["errors"]


def test_independent_parent_parameter_after_joint_metabolite_sentence_is_allowed() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = (
        "Following IM administration, Cmax for midazolam and its 1-hydroxy metabolite were "
        "approximately one-half of those achieved after intravenous injection. "
        "The volume of distribution (Vd) determined from six single-dose pharmacokinetic studies "
        "involving healthy adults ranged from 1.0-3.1 L/kg."
    )
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-independent-vd", "attempt_number": 1},
        input_hash="hash-independent-vd",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "clinical_pk",
        "parameter": "volume_of_distribution",
        "value": None,
        "value_low": 1.0,
        "value_high": 3.1,
        "unit": "L/kg",
        "condition": "healthy adults",
        "source_text": chunk["chunk_text"],
        "confidence": 0.72,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert len(valid) == 1
    assert rejected == []


def test_scalar_from_vs_comparison_is_rejected() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "Midazolam clearance was reduced (1.9 vs 2.8 mL/min/kg) in ARF patients."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-vs", "attempt_number": 1},
        input_hash="hash-vs",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "special_population",
        "parameter": "clearance",
        "value": 2.8,
        "unit": "mL/min/kg",
        "condition": "ARF patients",
        "source_text": chunk["chunk_text"],
        "confidence": 0.8,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert valid == []
    assert "ambiguous_comparative_value_assignment" in rejected[0]["errors"]


def test_parameter_unit_dimension_must_match() -> None:
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "clinical_pk",
        "parameter": "half_life",
        "value": 53,
        "unit": "ng/mL",
        "condition": "10 mg oral dose",
        "source_text": "After a 10 mg oral dose, midazolam had a half-life value of 53 ng/mL.",
        "confidence": 0.5,
        "requires_review": True,
    }
    chunk = _chunk()
    chunk["chunk_text"] = candidate["source_text"]
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-dimension", "attempt_number": 1},
        input_hash="hash-dimension",
    )

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert valid == []
    assert "unit_dimension_does_not_match_parameter" in rejected[0]["errors"]


def test_prefix_percent_change_requires_a_model_supplied_value() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "The AUC of midazolam increased by approximately 60% with food."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-prefix-percent", "attempt_number": 1},
        input_hash="hash-prefix-percent",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "clinical_pk",
        "parameter": "AUC",
        "value": None,
        "unit": None,
        "condition": "food",
        "source_text": chunk["chunk_text"],
        "confidence": 0.8,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert not valid
    assert "direct_percent_change_missing_value" in rejected[0]["errors"]
    candidate.update(value=60, unit="%")
    valid, rejected = _validate_candidates(task, [candidate], _upstream())
    assert not rejected and valid[0]["value"] == 60


def test_respectively_paired_values_are_not_assigned_as_scalar() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "The half-life of midazolam in blood and semen was 3.7 and 3.6 hours, respectively."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-respectively", "attempt_number": 1},
        input_hash="hash-respectively",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "clinical_pk",
        "parameter": "half_life",
        "value": 3.6,
        "unit": "hours",
        "condition": "semen",
        "source_text": chunk["chunk_text"],
        "confidence": 0.8,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert valid == []
    assert "ambiguous_comparative_value_assignment" in rejected[0]["errors"]


def test_dialysis_clearance_is_not_systemic_drug_clearance() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "Midazolam is dialyzable with a clearance of up to 170 mL/min during hemodialysis."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-dialysis", "attempt_number": 1},
        input_hash="hash-dialysis",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "clinical_pk",
        "parameter": "clearance",
        "value": 170,
        "unit": "mL/min",
        "condition": "hemodialysis",
        "source_text": chunk["chunk_text"],
        "confidence": 0.8,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert valid == []
    assert "dialysis_clearance_is_not_systemic_target_drug_clearance" in rejected[0]["errors"]


def test_creatinine_clearance_criterion_is_not_drug_clearance() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "Participants receiving midazolam required creatinine clearance above 60 mL/min."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-crcl", "attempt_number": 1},
        input_hash="hash-crcl",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "special_population",
        "parameter": "clearance",
        "value": 60,
        "unit": "mL/min",
        "condition": "eligibility criterion",
        "source_text": chunk["chunk_text"],
        "confidence": 0.5,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert valid == []
    assert "renal_function_clearance_is_not_target_drug_clearance" in rejected[0]["errors"]


def test_compound_treatment_expression_is_not_tissue_expression() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "Midazolam treatment increased beta receptor expression in adipose tissue."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-expression", "attempt_number": 1},
        input_hash="hash-expression",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "tissue_expression",
        "parameter": "beta_receptor_expression",
        "value": None,
        "unit": None,
        "condition": "adipose tissue",
        "source_text": chunk["chunk_text"],
        "confidence": 0.7,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert valid == []
    assert "tissue_expression_requires_structured_expression_source" in rejected[0]["errors"]


def test_generic_population_pk_model_is_not_public_pbpk_model() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "A population pharmacokinetic model for midazolam was developed using sparse samples."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-pop-pk", "attempt_number": 1},
        input_hash="hash-pop-pk",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "public_pbpk_model",
        "parameter": "model_development",
        "value": None,
        "unit": None,
        "condition": "population PK",
        "source_text": chunk["chunk_text"],
        "confidence": 0.7,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert valid == []
    assert "public_pbpk_model_lacks_explicit_pbpk_marker" in rejected[0]["errors"]


def test_explicit_pbpk_model_candidate_is_allowed() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "A whole-body PBPK model for midazolam was developed in PK-Sim."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-pbpk", "attempt_number": 1},
        input_hash="hash-pbpk",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "public_pbpk_model",
        "parameter": "model_development",
        "value": None,
        "unit": None,
        "condition": "PK-Sim",
        "source_text": chunk["chunk_text"],
        "confidence": 0.9,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert len(valid) == 1
    assert rejected == []


def test_special_population_marker_must_be_in_source_text() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "Renal clearance is a minor determinant of response to midazolam."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-special-marker", "attempt_number": 1},
        input_hash="hash-special-marker",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "special_population",
        "parameter": "renal_clearance",
        "value": None,
        "unit": None,
        "condition": "renal impairment",
        "source_text": chunk["chunk_text"],
        "confidence": 0.6,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert valid == []
    assert "special_population_marker_missing_from_source_text" in rejected[0]["errors"]


def test_special_population_marker_in_source_is_allowed() -> None:
    chunk = _chunk()
    chunk["chunk_text"] = "The half-life of midazolam was prolonged in patients with renal impairment."
    task = PendingTask(
        chunk=chunk,
        target_compound="midazolam",
        status_record={},
        request_record={"request_id": "request-special-valid", "attempt_number": 1},
        input_hash="hash-special-valid",
    )
    candidate = {
        "target_compound": "midazolam",
        "evidence_type": "special_population",
        "parameter": "half_life",
        "value": None,
        "unit": None,
        "condition": "renal impairment",
        "source_text": chunk["chunk_text"],
        "confidence": 0.8,
        "requires_review": True,
    }

    valid, rejected = _validate_candidates(task, [candidate], _upstream())

    assert len(valid) == 1
    assert rejected == []


def test_shared_chunk_resolves_only_one_exact_target() -> None:
    chunk = {"compound": None, "chunk_text": "A public PBPK model for warfarin was evaluated."}
    target, status = _resolve_target_compound(chunk, {"midazolam", "warfarin"})

    assert target == "warfarin"
    assert status == "unique_exact_target_in_chunk_text"


def test_shared_chunk_with_multiple_targets_is_unresolved() -> None:
    chunk = {"compound": "shared", "chunk_text": "Midazolam and warfarin were listed together."}
    target, status = _resolve_target_compound(chunk, {"midazolam", "warfarin"})

    assert target is None
    assert status == "multiple_target_compounds_in_chunk_text"


def test_api_failure_is_resolved_when_request_is_no_longer_eligible(tmp_path) -> None:
    path = tmp_path / "processing_failures.jsonl"
    rows = [
        {
            "stage": "llm_extract",
            "failure_type": "llm_api_failure",
            "request_id": "obsolete-request",
            "resolved": False,
        },
        {
            "stage": "llm_extract",
            "failure_type": "llm_api_failure",
            "request_id": "eligible-request",
            "resolved": False,
        },
        {
            "stage": "llm_extract",
            "failure_type": "llm_candidate_validation_rejection",
            "request_id": "obsolete-request",
            "resolved": True,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    _resolve_ineligible_api_failures(path, {"eligible-request"})

    updated = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert updated[0]["resolved"] is True
    assert updated[0]["resolution"] == "no_longer_eligible_under_current_routing"
    assert updated[1]["resolved"] is False
    assert updated[2]["failure_type"] == "llm_candidate_validation_rejection"


def _validation_case(source, **changes):
    chunk = _chunk()
    chunk["chunk_text"] = source
    task = PendingTask(chunk, "midazolam", {}, {"request_id": "regression", "input_items": [source]}, "input")
    candidate = {
        "target_compound": "midazolam", "evidence_type": "clinical_pk", "parameter": "Tmax",
        "value": None, "value_low": None, "value_high": None, "value_qualifier": None,
        "unit": None, "condition": "adults", "source_text": source, "confidence": .7, "requires_review": True,
        **changes,
    }
    return _validate_candidates(task, [candidate], _upstream())


def test_mixed_time_range_requires_both_correct_parameter_bound_conversions():
    source = "Midazolam time to reach peak concentration (T max ) ranged from 30 minutes to 2 hours."
    valid, rejected = _validation_case(source, value_low=.5, value_high=2, unit="hours")
    assert not rejected and len(valid) == 1
    assert "verified_source_time_range_unit_conversion" in valid[0]["validation_repairs"]
    assert valid[0]["source_text"] == source
    valid, rejected = _validation_case(source, value_low=.6, value_high=2, unit="hours")
    assert not valid and "numeric_value_not_found_in_source_text" in rejected[0]["errors"]


def test_time_conversion_cannot_borrow_dosing_interval_or_other_parameter():
    for source in (
        "Midazolam was infused from 30 minutes to 2 hours; Tmax was measured in adults.",
        "Midazolam half-life ranged from 30 minutes to 2 hours; Tmax was measured in adults.",
    ):
        valid, rejected = _validation_case(source, value_low=.5, value_high=2, unit="hours")
        assert not valid and "numeric_value_not_found_in_source_text" in rejected[0]["errors"]


def test_percent_interval_connector_does_not_change_true_negative_signs():
    from pbpk_kb.processing.run_llm_extraction import _number_appears_in_source
    assert _number_appears_in_source(102, "87.9%-102%")
    assert not _number_appears_in_source(-102, "87.9%-102%")
    assert _number_appears_in_source(-102, "The change was -102%.")
    assert not _number_appears_in_source(102, "The change was -102%.")
    assert not _number_appears_in_source(39, "390")
    source = "Midazolam AUC was unchanged (90% confidence interval: 87.9%-102%) with an inhibitor."
    valid, rejected = _validation_case(source, evidence_type="ddi", parameter="AUC",
                                      value_low=87.9, value_high=102, unit="%", value_qualifier="90% confidence interval")
    assert len(valid) == 1 and not rejected


def test_soft_hyphen_repair_restores_only_the_complete_original_quote():
    source = "Midazolam volume of distribution in infants was high\u00ader than in adults (0.6 L/kg)."
    valid, rejected = _validation_case(source, parameter="volume_of_distribution", value=.6, unit="L/kg",
                                      source_text=source.replace("\u00ad", ""))
    assert not rejected and valid[0]["source_text"] == source
    assert "restored_verbatim_source_soft_hyphen" in valid[0]["validation_repairs"]
    valid, rejected = _validation_case(source, parameter="volume_of_distribution", value=.6, unit="L/kg",
                                      source_text=source.replace("\u00ad", "").replace("0.6", "0.7"))
    assert not valid and "source_text_not_found_verbatim_in_chunk" in rejected[0]["errors"]


def test_mixed_scalar_and_range_fields_still_rejected():
    source = "Midazolam volume of distribution in infants ranged from 0.8 to 0.9 L/kg."
    valid, rejected = _validation_case(source, parameter="volume_of_distribution",
                                      value=.8, value_low=.9, unit="L/kg")
    assert not valid and "candidate_cannot_have_scalar_and_range_values" in rejected[0]["errors"]


def test_population_excretion_qualitative_assertion_has_no_numeric_qualifier():
    source = "The fraction excreted unchanged in urine of midazolam in infants was inversely related to age."
    valid, rejected = _validation_case(source, parameter="fraction excreted unchanged in urine",
                                      evidence_type="special_population", condition="infants, inversely related to age")
    assert len(valid) == 1 and not rejected
    valid, rejected = _validation_case(source, parameter="fraction excreted unchanged in urine",
                                      evidence_type="special_population", value_qualifier="inversely related")
    assert not rejected and valid[0]["value_qualifier"] is None
    assert "inversely related" in valid[0]["condition"]
    assert "source_supported_description_moved_to_condition" in valid[0]["validation_repairs"]


def test_response_reuse_requires_current_validator_and_prompt(tmp_path):
    from pbpk_kb.processing.run_llm_extraction import (
        VALIDATOR_VERSION, _load_processed_responses, _load_valid_candidates_by_input_hash,
    )
    version = LLMExtractionAgent(mode="off").prompt_template_version
    base = {"status": "ok", "candidates": [], "prompt_template_version": version,
            "validator_version": VALIDATOR_VERSION}
    rows = [
        {**base, "request_id": "current", "input_hash": "current"},
        {**base, "request_id": "stale-validator", "input_hash": "stale-validator", "validator_version": "old"},
        {**base, "request_id": "stale-prompt", "input_hash": "stale-prompt", "prompt_template_version": "old"},
    ]
    path = tmp_path / "responses.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert set(_load_processed_responses(path, version)) == {"current"}
    assert set(_load_valid_candidates_by_input_hash(path, version)) == {"current"}


def test_qualifier_repair_requires_literal_source_support_and_keeps_inequalities():
    source = "Midazolam half-life in infants was related to age."
    valid, rejected = _validation_case(source, parameter="half_life", value_qualifier="inversely related")
    assert not valid and "qualitative_candidate_must_not_have_value_qualifier" in rejected[0]["errors"]
    source = "Less than 0.5% of midazolam was excreted unchanged in urine."
    valid, rejected = _validation_case(source, parameter="excretion", evidence_type="adme",
                                      value=.5, unit="%", value_qualifier=">")
    assert not valid and "value_qualifier_does_not_match_source_bound" in rejected[0]["errors"]


def test_potential_interaction_is_not_rewritten_as_observed_interaction():
    source = "Midazolam has the potential to interact with drugs that inhibit CYP3A4."
    valid, rejected = _validation_case(source, parameter="enzyme_interaction", evidence_type="ddi",
                                      condition="interacts with CYP3A4 inhibitors")
    assert not rejected
    assert valid[0]["condition"].startswith("Potential interaction;")


def test_common_qualitative_descriptions_move_to_condition_without_creating_numbers():
    for description in ("increase", "decrease", "significantly higher", "significant increase", "unchanged"):
        source = f"Midazolam clearance showed a {description} with an inhibitor."
        valid, rejected = _validation_case(source, evidence_type="ddi", parameter="clearance",
                                          value_qualifier=description)
        assert not rejected and valid[0]["value_qualifier"] is None
        assert description in valid[0]["condition"]
        assert valid[0]["value"] is None and valid[0]["unit"] is None
    source = "Midazolam clearance was higher than 3 L/h."
    valid, rejected = _validation_case(source, parameter="clearance", value=3, unit="L/h", value_qualifier="higher than")
    assert not rejected and valid[0]["value_qualifier"] == "higher than"


def test_plural_auc_and_typeset_cmax_terminology_are_recognized():
    for parameter, text in (("AUC", "Midazolam AUCs increased with an inhibitor."),
                            ("Cmax", "Midazolam C max increased with an inhibitor.")):
        valid, rejected = _validation_case(text, parameter=parameter, evidence_type="ddi")
        assert len(valid) == 1 and not rejected


def test_truncated_endpoint_list_cannot_share_first_percent_change():
    source = "Ketoconazole increased the AUC 24h, C max and C min of midazolam by 22%"
    for parameter in ("AUC 24h", "Cmax", "Cmin"):
        valid, rejected = _validation_case(source, parameter=parameter, evidence_type="ddi", value=22, unit="%")
        assert not valid
        assert "truncated_multi_endpoint_percent_assignment" in rejected[0]["errors"]


def test_complete_or_explicit_shared_percent_statement_remains_supported():
    from pbpk_kb.processing.run_llm_extraction import _numeric_parameter_owner_error
    for source in (
        "Ketoconazole increased the AUC and Cmax of midazolam by 22%.",
        "Ketoconazole increased both AUC and Cmax of midazolam by 22%",
        "Midazolam AUC increased by 22%",
        "Midazolam Cmax increased by 9%; AUC increased by 22%",
    ):
        assert _numeric_parameter_owner_error(dict(parameter="AUC", value=22, unit="%", source_text=source), "midazolam") is None


def test_joint_parent_and_named_chemical_excretion_is_not_assigned_to_parent():
    source = "Urinary excretion of midazolam and 5-acetylamino-6-formylamino-3-methyluracil was unchanged."
    valid, rejected = _validation_case(source, parameter="urinary_excretion", evidence_type="adme")
    assert not valid and "joint_parent_and_metabolite_subject" in rejected[0]["errors"]


def test_other_drug_plasma_pk_cannot_be_assigned_to_coadministered_target():
    text = "The Cmax, Tmax and AUC of plasma salicylate did not differ when aspirin was given alone or with midazolam."
    for parameter in ("Cmax", "Tmax", "AUC"):
        valid, rejected = _validation_case(text, parameter=parameter)
        assert not valid and "parameter_explicitly_owned_by_other_substance" in rejected[0]["errors"]


def test_metabolite_excretion_cannot_be_assigned_to_parent_mentioned_only_as_origin():
    text = "Administration of this metabolite showed that the acid was excreted unchanged and was an end product of midazolam metabolism."
    valid, rejected = _validation_case(text, parameter="excretion", evidence_type="adme")
    assert not valid and "metabolite_parameter_is_not_parent_drug_parameter" in rejected[0]["errors"]
    text = "The metabolite of midazolam had a half-life of 3 hours."
    valid, rejected = _validation_case(text, parameter="half_life", value=3, unit="hours")
    assert not valid and "metabolite_parameter_is_not_parent_drug_parameter" in rejected[0]["errors"]


def test_separate_explicit_parent_assertion_is_kept_despite_metabolite_context():
    text = "Midazolam was excreted unchanged in urine; its metabolites were measured separately."
    valid, rejected = _validation_case(text, parameter="excretion", evidence_type="adme")
    assert valid and not rejected
    text = "Midazolam was metabolized into several metabolites by CYP3A4."
    valid, rejected = _validation_case(text, parameter="metabolism", evidence_type="adme")
    assert valid and not rejected



def _binding_validation(source, *, target="midazolam", quoted=None, value=None, unit=None, **changes):
    chunk = _chunk()
    chunk.update(compound=target, chunk_text=source)
    task = PendingTask(chunk, target, {}, {"request_id": "binding-owner", "input_items": [source]}, "binding-input")
    candidate = {
        "target_compound": target, "evidence_type": "adme", "parameter": "protein_binding",
        "value": value, "unit": unit, "condition": None, "source_text": quoted or source,
        "confidence": 0.8, "requires_review": True, **changes,
    }
    return _validate_candidates(task, [candidate], _upstream())


def test_bounded_percent_repair_does_not_borrow_comparator_drug_class_value():
    source = (
        "Distribution The apparent volume of distribution (V/F) of Metformin following single oral doses "
        "of Metformin Hydrochloride Tablets USP 850 mg averaged 654 ± 358 L. "
        "Metformin is negligibly bound to plasma proteins, in contrast to sulfonylureas, "
        "which are more than 90% protein bound."
    )
    valid, rejected = _binding_validation(
        source, target="metformin", quoted="Metformin is negligibly bound to plasma proteins",
    )
    assert not rejected and len(valid) == 1
    assert valid[0]["value"] is None and valid[0]["unit"] is None
    assert valid[0].get("value_qualifier") is None
    assert valid[0]["source_text"] == source
    assert "bounded_percent_from_source_text" not in valid[0].get("validation_repairs", [])
    valid, rejected = _binding_validation(source, target="metformin", value=90, unit="%", value_qualifier=">")
    assert not valid
    assert "numeric_value_belongs_to_other_substance" in rejected[0]["errors"]


@pytest.mark.parametrize("source,value", [
    ("Midazolam is negligibly bound to plasma proteins, in contrast to anticonvulsants, which are more than 85% protein bound.", 85),
    ("Midazolam is negligibly bound to plasma proteins, whereas Comparatorin is more than 96% protein bound.", 96),
    ("Comparatorin is more than 96% protein bound, whereas Midazolam is negligibly bound to plasma proteins.", 96),
    ("More than 96% of Comparatorin is bound to plasma proteins, whereas Midazolam is negligibly bound to plasma proteins.", 96),
])
def test_other_substance_binding_value_is_never_assigned_to_target(source, value):
    from pbpk_kb.processing.run_llm_extraction import _numeric_parameter_owner_error
    assert _numeric_parameter_owner_error(
        {"parameter": "protein_binding", "source_text": source, "value": value, "unit": "%"},
        "midazolam",
    ) == "numeric_value_belongs_to_other_substance"
    valid, rejected = _binding_validation(source)
    assert not rejected and valid[0]["value"] is None


@pytest.mark.parametrize("source,value", [
    ("Midazolam is more than 95% bound to plasma proteins.", 95),
    ("After a 90 mg oral dose, Midazolam is more than 95% protein-bound.", 95),
    ("Midazolam has a volume of distribution of 75 L. Midazolam is more than 95% bound to plasma proteins.", 95),
    ("Unlike comparator drugs, Midazolam is more than 95% protein bound.", 95),
    ("Midazolam is more than 95% protein bound in healthy adults, while patients with renal impairment have 85% protein binding.", 95),
])
def test_explicit_target_binding_rate_keeps_its_value_with_other_context_numbers(source, value):
    valid, rejected = _binding_validation(source, value=value, unit="%")
    assert not rejected and valid[0]["value"] == value
    assert valid[0]["value_qualifier"] == ">"


@pytest.mark.parametrize("value", [90, 85])
def test_same_drug_two_population_values_are_not_other_drug_ownership(value):
    from pbpk_kb.processing.run_llm_extraction import _numeric_parameter_owner_error
    source = ("Midazolam is 90% protein bound in healthy adults, while patients with renal impairment "
              "have 85% protein binding.")
    assert _numeric_parameter_owner_error(
        SimpleNamespace(parameter="protein_binding", source_text=source, value=value, unit="%"),
        "midazolam",
    ) is None


def test_binding_does_not_borrow_a_bioavailability_percentage_or_dose_number():
    source = ("Midazolam bioavailability is greater than 85%, and protein binding is negligible "
              "after a 90 mg dose.")
    valid, rejected = _binding_validation(source)
    assert not rejected and valid[0]["value"] is None
    valid, rejected = _binding_validation(source, value=85, unit="%")
    assert not valid
    assert "numeric_value_not_supported_for_target_parameter" in rejected[0]["errors"]
    valid, rejected = _binding_validation("After a 90 mg dose, Midazolam is negligibly bound to plasma proteins.", value=90, unit="%")
    assert not valid
    assert "numeric_value_not_supported_for_target_parameter" in rejected[0]["errors"]


def test_target_qualitative_binding_without_comparator_is_retained():
    valid, rejected = _binding_validation("Midazolam is negligibly bound to plasma proteins.")
    assert not rejected and valid[0]["value"] is None and valid[0]["unit"] is None


def test_target_binding_ascii_range_endpoints_are_preserved():
    valid, rejected = _binding_validation(
        "Midazolam plasma protein binding ranged from 60-70%.",
        value_low=60, value_high=70, unit="%",
    )
    assert not rejected and valid[0]["value_low"] == 60 and valid[0]["value_high"] == 70



def test_real_caffeine_half_life_does_not_treat_serum_and_saliva_as_drug_subjects():
    source = ("The half-life of caffeine in both serum and saliva was approximately 3 hr, with the "
              "concentration of caffeine in the saliva samples ranging from 65 to 85% of that found in the serum samples.")
    valid, rejected = _binding_validation(
        source, target="caffeine", parameter="half_life", evidence_type="clinical_pk", value=3, unit="hr",
    )
    assert not rejected and valid[0]["value"] == 3 and valid[0]["unit"] == "hr"


@pytest.mark.parametrize("parameter", ["Cmax", "AUC"])
@pytest.mark.parametrize("prefix", ["", "to body surface area of 1.73 m2 Pediatrics After "])
def test_real_metformin_coordinated_parameters_share_explicit_five_percent_value(parameter, prefix):
    source = (prefix + "administration of a single oral Metformin Hydrochloride 500 mg tablet with food, "
              "geometric mean metformin Cmax and AUC differed less than 5% between pediatric type 2 diabetic "
              "patients (12 to 16 years of age) and gender- and weight-matched healthy adults (20 to 45 years of age), "
              "all with normal renal function.")
    valid, rejected = _binding_validation(
        source, target="metformin", parameter=parameter, evidence_type="special_population",
        value=5, unit="%", value_qualifier="<",
    )
    assert not rejected and valid[0]["value"] == 5


def test_real_caffeine_metabolism_never_autofills_remaining_parent_fraction():
    source = ("Caffeine is rapidly metabolized in human and mouse cells in culture by de-methylation: within 1-3 h "
              "of exposure to millimolar concentrations of labeled caffeine, more than 90% of the pool consists "
              "of labeled products of metabolism and less than 10% is still caffeine.")
    valid, rejected = _binding_validation(source, target="caffeine", parameter="metabolism")
    assert not rejected and valid[0]["value"] is None and valid[0]["unit"] is None
    assert "bounded_percent_from_source_text" not in valid[0].get("validation_repairs", [])
    valid, rejected = _binding_validation(
        source, target="caffeine", parameter="metabolism", value=10, unit="%", value_qualifier="<",
    )
    assert not valid
    assert "numeric_value_not_supported_for_target_parameter" in rejected[0]["errors"]


def test_real_relative_half_life_percent_range_cannot_be_labeled_hours():
    from pbpk_kb.processing.run_llm_extraction import _numeric_range_source_error
    source = ("In the rat, single dose Danshen aqueous extract treatment (100 or 200 mg/kg, i.p.) decreased metabolism "
              "of caffeine to paraxanthine, with overall decrease in caffeine clearance (6-20%), increase in area "
              "under the curve (AUC; 7-24%) and plasma half-life (t(1/2) 14-16%).")
    valid, rejected = _binding_validation(
        source, target="caffeine", parameter="half_life", evidence_type="clinical_pk",
        value_low=14, value_high=16, unit="h",
    )
    assert not valid
    assert "numeric_unit_conflicts_with_source_assertion" in rejected[0]["errors"]
    assert _numeric_range_source_error(SimpleNamespace(
        parameter="half_life", source_text=source, value=None, value_low=14, value_high=16, unit="h",
    )) == "range_unit_conflicts_with_source_interval"
    valid, rejected = _binding_validation(
        source, target="caffeine", parameter="half_life", evidence_type="clinical_pk",
        value_low=14, value_high=16, unit="%",
    )
    assert not rejected and valid[0]["unit"] == "%"



def test_percent_qualifier_cannot_be_attached_to_same_number_in_hours():
    source = ("The half-life of caffeine was 3 hr, with less than 3% of caffeine recovered in saliva.")
    valid, rejected = _binding_validation(
        source, target="caffeine", parameter="half_life", evidence_type="clinical_pk", value=3, unit="hr",
    )
    assert not rejected and valid[0]["value"] == 3
    assert valid[0].get("value_qualifier") is None


@pytest.mark.parametrize("changed", [None, "input_hash", "validator_version", "prompt_template_version", "status"])
def test_resume_settles_failures_only_after_matching_current_success(tmp_path, monkeypatch, changed):
    from collections import Counter
    from pbpk_kb.processing import run_llm_extraction as module
    from pbpk_kb.processing.chunk_classifier import prepare_llm_input
    from pbpk_kb.processing.raw_processing_paths import raw_processing_layout

    layout = raw_processing_layout(tmp_path)
    layout.ensure_dirs()
    chunk = _chunk()
    def write_rows(name, rows):
        layout.path(name, for_write=True).write_text("".join(json.dumps(r) + "\n" for r in rows))
    layout.path("pipeline_state.json", for_write=True).write_text(json.dumps({
        "run_id": "run-test", "stages": {
            name: {"status": "completed", "accepted": True}
            for name in ("inventory", "parse", "deterministic_extract")
        },
    }))
    write_rows("raw_inventory.jsonl", [{**_upstream().raw_assets["asset-1"], "compound": "midazolam"}])
    write_rows("parsed_documents.jsonl", [{"parsed_document_id": "doc-1", "source_id": "pmc_oa"}])
    write_rows("parsed_tables.jsonl", [])
    write_rows("text_chunks.jsonl", [chunk])
    write_rows("evidence_candidates_deterministic.jsonl", [])
    agent = LLMExtractionAgent(mode="live")
    request = module._request_record(agent, chunk, "midazolam", Counter(), prepare_llm_input(chunk, "midazolam"))
    request["attempt_number"] = 3
    response = module._response_record(
        PendingTask(chunk, "midazolam", {}, request, request["input_hash"]),
        status="ok", model_name=agent.model_name, prompt_template_version=agent.prompt_template_version,
        candidates=[], rejected_candidates=[], schema_validation_status="valid_empty",
        error=None, latency_seconds=1., output_hash="real-empty-success",
    )
    if changed is not None:
        response[changed] = "failed_local_llm" if changed == "status" else "different"
    write_rows("llm_extraction_requests_live.jsonl", [request])
    write_rows("llm_extraction_responses_live.jsonl", [response])
    failures = [{"failure_id": str(n), "stage": "llm_extract", "failure_type": "llm_api_failure",
                 "request_id": request["request_id"], "resolved": False, "reason": "historical GPU failure"}
                for n in (1, 2)]
    write_rows("processing_failures.jsonl", failures)
    def no_repeat(*args, **kwargs):
        raise AssertionError("A preserved successful input must not trigger another model call")
    monkeypatch.setattr(LLMExtractionAgent, "extract", no_repeat)
    summary = module.run_llm_extraction(
        processed_dir=tmp_path, llm_max_requests=0, min_chars=1, fail_fast=True,
        classify_only=False, concurrency=1,
    )
    rows = [json.loads(line) for line in layout.path("processing_failures.jsonl").read_text().splitlines()]
    assert summary["accepted"] is (changed is None)
    assert summary["api_request_count"] == 0
    assert len(rows) == 2
    for before, after in zip(failures, rows):
        assert all(after[k] == v for k, v in before.items() if k != "resolved")
        assert after["resolved"] is (changed is None)
        if changed is None:
            assert after["resolution"] == "later_attempt_succeeded"
