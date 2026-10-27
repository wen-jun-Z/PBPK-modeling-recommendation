from pathlib import Path

import json

import pytest

from pbpk_kb.processing import auto_quality_llm_stage as quality
from pbpk_kb.processing.auto_quality_llm_stage import ContextIndex, parse_json_object


def test_parse_json_object_accepts_fenced_surrounding_text() -> None:
    assert parse_json_object('result: {"supported": true}') == {"supported": True}


def test_context_index_returns_adjacent_chunks(tmp_path: Path) -> None:
    folder = tmp_path / "03_chunks_and_embeddings"
    folder.mkdir(parents=True)
    path = folder / "text_chunks.jsonl"
    path.write_text(
        "\n".join(
            f'{{"chunk_id":"c{i}","parsed_document_id":"d1","chunk_index":{i},"chunk_text":"text {i}","source_location":"s{i}"}}'
            for i in range(3)
        ) + "\n",
        encoding="utf-8",
    )

    context = ContextIndex(tmp_path).context_for({"chunk_id": "c1"})

    assert [row["chunk_id"] for row in context["chunks"]] == ["c0", "c1", "c2"]
    assert context["combined_text"] == "text 0\n\ntext 1\n\ntext 2"


def validation_response(quote: str, **changes) -> dict:
    return {"supported": True, "unsupported_fields": [], "compound_role_correct": True,
            "parameter_value_unit_correct": True, "reference_correct": True,
            "context_sufficient": True, "verbatim_support": quote, "confidence": .9,
            "reason": "Each field is stated in the source.", **changes}


def lmfe_accepts(schema: dict, response: dict) -> bool:
    from lmformatenforcer import JsonSchemaParser
    parser = JsonSchemaParser(schema)
    for character in json.dumps(response, separators=(",", ":")):
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


def test_quality_reason_bound_is_enforced_by_validator_and_registered_decoder():
    from jsonschema import Draft202012Validator

    schema = quality.quality_output_schema(quality.VALIDATION_PROMPT, {})
    validator = Draft202012Validator(schema)
    for reason, expected in [("x" * 512, True), ("x" * 513, False), ("", False)]:
        response = validation_response("Clearance is 404 ml/min.", reason=reason)
        assert validator.is_valid(response) == expected
        assert lmfe_accepts(schema, response) == expected
    assert not validator.is_valid(validation_response("Clearance is 404 ml/min.", reason=""))
    rejected = validation_response("No mean or unit was stated.", supported=False,
                                  parameter_value_unit_correct=False, unsupported_fields=["value_qualifier", "unit"])
    assert validator.is_valid(rejected)
    assert lmfe_accepts(schema, rejected)


def test_quality_output_text_limits_keep_multiline_quotes_and_reject_long_explanations():
    from jsonschema import Draft202012Validator
    from lmformatenforcer import JsonSchemaParser

    quote = 'Mean clearance\nwas 404 "ml/min"; literal slash \\ preserved.'
    for prompt, payload, response, oversized in [
        (quality.REPAIR_PROMPT, {"evidence_id": "e1"},
         {"evidence_id": "e1", "repairs": {"dose": 600, "condition": "steady state"}, "supporting_text": quote, "confidence": .9},
         {"repairs": {"condition": "x" * 1025}}),
        (quality.VALIDATION_PROMPT, {}, validation_response(quote), {"unsupported_fields": ["x" * 129]}),
    ]:
        schema = quality.quality_output_schema(prompt, payload)
        validator = Draft202012Validator(schema)
        JsonSchemaParser(schema)  # Reject unsupported pattern/length combinations before HTTP use.
        assert validator.is_valid(response)
        assert lmfe_accepts(schema, response)  # Escaped quotes, backslash and newline remain generatable.
        assert not validator.is_valid({**response, **oversized})
        quote_key = "supporting_text" if prompt == quality.REPAIR_PROMPT else "verbatim_support"
        assert not validator.is_valid({**response, quote_key: "x" * 4097})


@pytest.mark.parametrize("field", ["reason", "verbatim_support", "supporting_text", "condition", "unsupported_field"])
def test_all_quality_text_limits_keep_lmfe_fast_path_and_close_at_bound(field):
    from lmformatenforcer import JsonSchemaParser
    validation = quality.quality_output_schema(quality.VALIDATION_PROMPT, {})["properties"]
    repair = quality.quality_output_schema(quality.REPAIR_PROMPT, {"evidence_id": "e1"})["properties"]
    schemas = {"reason": validation["reason"], "verbatim_support": validation["verbatim_support"],
               "supporting_text": repair["supporting_text"], "condition": repair["repairs"]["properties"]["condition"],
               "unsupported_field": validation["unsupported_fields"]["items"]}
    schema = {**schemas[field], "type": "string"}
    parser = JsonSchemaParser(schema).add_character('"')
    assert parser.shortcut_key()[0] == "json_freetext"
    for _ in range(schema["maxLength"]):
        assert "x" in parser.get_allowed_characters()
        parser = parser.add_character("x")
    assert "x" not in parser.get_allowed_characters()
    assert '"' in parser.get_allowed_characters()


def test_multiline_reason_is_rejected_after_decoding_without_restricting_source_quotes(monkeypatch):
    quote = "The mean clearance\nwas 404 ml/min."
    monkeypatch.setattr(quality, "generate_evidence_json", lambda *args: validation_response(quote, reason="Line one.\nLine two."))
    stage = object.__new__(quality.AutoQualityLLMStage)
    with pytest.raises(quality.QualityOutputError, match="single line"):
        stage._chat(quality.VALIDATION_PROMPT, {})


@pytest.mark.parametrize("qualifier,quote,rejected", [
    ("mean", "Total clearance 404 vs. 314 ml/min.", True),
    ("median", "The mean clearance was 404 ml/min.", True),
    ("mean", "The mean residence time was 2 h; clearance was 404 ml/min.", True),
    ("mean", "The mean clearance was 404 ml/min.", False),
    ("geometric_mean", "The geometric mean clearance was 404 ml/min.", False),
    (None, "Total clearance was 404 ml/min.", False),
])
def test_reported_numbers_do_not_imply_a_statistical_role(qualifier, quote, rejected):
    result = quality.apply_validation_source_guards(
        validation_response(quote), {"value": 404, "value_qualifier": qualifier}, quote)
    assert result["supported"] is (not rejected)
    assert result["deterministic_statistic_failures"] == (["value_qualifier"] if rejected else [])


def test_comparison_cannot_become_an_interval_even_if_both_numbers_occur():
    quote = "With or without plasmapheresis, total clearance was 404 vs. 314 ml/min."
    fields = {"value": None, "value_low": 314, "value_high": 404, "parameter": "clearance",
              "source_text": quote, "unit": "ml/min", "value_qualifier": "mean"}
    result = quality.apply_validation_source_guards(validation_response(quote), fields, quote)
    assert result["supported"] is False
    assert set(result["unsupported_fields"]) == {"value_low", "value_high", "value_qualifier"}


def test_validation_chat_applies_statistical_role_guard(monkeypatch):
    quote = "Total clearance was 404 ml/min."
    monkeypatch.setattr(quality, "generate_evidence_json", lambda *args: validation_response(quote))
    stage = object.__new__(quality.AutoQualityLLMStage)
    result = stage._chat(quality.VALIDATION_PROMPT, {
        "original_evidence": {"current_fields": {"value": 404}, "source_text": quote},
        "repaired_fields": {"value_qualifier": "mean"},
        "source_context": {"combined_text": quote},
    })
    assert result["supported"] is False
    assert result["unsupported_fields"] == ["value_qualifier"]
    assert result["generation_audit"]["prompt_sha256"]


@pytest.mark.parametrize("failure", ["missing_context", "repair_quote", "validation_quote", "number", "statistic"])
def test_cached_repair_promotion_cannot_bypass_source_checks(monkeypatch, failure):
    quote = "Total clearance was 404 ml/min."
    row = {"evidence_id": "e1", "source_text": quote, "value": None, "parameter": "clearance"}
    repairs = {"value": 390 if failure == "number" else 404, "unit": "ml/min"}
    if failure == "statistic":
        repairs["value_qualifier"] = "mean"
    result = {
        "repair_response": {"repairs": repairs, "supporting_text": "Invented." if failure == "repair_quote" else quote},
        "validation_response": validation_response("Invented." if failure == "validation_quote" else quote),
    }
    def unexpected_rule(*args):
        raise AssertionError("Unsupported cached repair reached rule promotion")
    monkeypatch.setattr(quality, "evaluate_evidence", unexpected_rule)
    context = None if failure == "missing_context" else {"combined_text": quote}
    assert quality.promote_from_existing_validation(row, result, context=context) is None


def test_cached_supported_repair_can_still_pass_rule_recheck(monkeypatch):
    quote = "The mean clearance was 404 ml/min."
    row = {"evidence_id": "e1", "source_text": quote, "value": None, "parameter": "clearance"}
    result = {
        "repair_response": {"repairs": {"value": 404, "unit": "ml/min", "value_qualifier": "mean"}, "supporting_text": quote},
        "validation_response": validation_response(quote),
    }
    monkeypatch.setattr(quality, "deterministic_repair", lambda row: (row, []))
    monkeypatch.setattr(quality, "evaluate_evidence", lambda *args: {"quality_status": "auto_validated"})
    promoted = quality.promote_from_existing_validation(row, result, context={"combined_text": quote})
    assert promoted["status"] == "auto_validated"
    assert promoted["evidence"]["llm_validation"]["deterministic_statistic_failures"] == []


@pytest.mark.parametrize("use_cache", [False, True])
def test_repaired_numeric_value_recomputes_derived_normalization(tmp_path, monkeypatch, use_cache):
    import hashlib
    text = "After dosing, caffeine Cmax was 10 ng/mL."
    raw = tmp_path / "source.txt"; raw.write_text(text)
    row = {"evidence_id": "e1", "compound_id": "c1", "compound_name": "caffeine", "compound_role": "target_compound",
           "evidence_type": "clinical_pk", "parameter": "Cmax", "value": 20, "unit": "ng/mL",
           "canonical_value": 999, "canonical_unit": "mg/L", "source_id": "pubmed", "raw_asset_id": "r1",
           "storage_path": str(raw), "checksum": hashlib.sha256(raw.read_bytes()).hexdigest(),
           "source_location": "pubmed:1#abstract", "source_text": text, "source_text_hash": hashlib.sha256(text.encode()).hexdigest(),
           "extraction_method": "rule_based", "extractor_version": "1"}
    repair = {"repairs": {"value": 10, "unit": "ng/mL"}, "supporting_text": text, "confidence": .9}
    outputs = iter([repair, validation_response(text)])
    stage = object.__new__(quality.AutoQualityLLMStage); stage.model = "mock"
    monkeypatch.setattr(stage, "_chat", lambda *args: next(outputs))
    result = (quality.promote_from_existing_validation(row, {"repair_response": repair, "validation_response": validation_response(text)}, context={"combined_text": text})
              if use_cache else stage._process_one(row, {"combined_text": text}))
    assert result["status"] == "auto_validated"
    assert result["evidence"]["value"] == result["evidence"]["canonical_value"] == 10
    assert result["evidence"]["canonical_unit"] == "ng/mL"
    assert row["canonical_value"] == 999
