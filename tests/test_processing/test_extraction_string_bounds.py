"""Extraction decoder bounds, exercised against project LMFE without a GPU."""
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from pbpk_kb.processing import llm_agent
from pbpk_kb.processing.run_llm_extraction import _unit_appears_in_source


QUOTE = "The initial volume of distribution of paclitaxel was 3,14 ± 0,85 l × kg(-1)."
PAYLOAD = {"target_compound": "paclitaxel", "input_items": [QUOTE], "input_text_or_table": QUOTE}
ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("bounded_extraction_server", ROOT / "scripts/deployment/evidence_qwen_server.py")
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def candidate(**updates):
    return {"target_compound": "paclitaxel", "evidence_type": "clinical_pk",
            "parameter": "volume_of_distribution", "value": 3.14,
            "value_low": None, "value_high": None, "value_qualifier": None,
            "unit": "l × kg(-1)", "condition": None, "source_item_index": 0,
            "confidence": .9, "requires_review": True, **updates}


def consume(schema, text):
    from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
    parser = JsonSchemaParser(schema, config=CharacterLevelParserConfig(
        alphabet="".join(chr(i) for i in range(32, 127)) + "×±µ中"))
    for character in text:
        if character not in parser.get_allowed_characters():
            return None
        parser = parser.add_character(character)
    return parser


@pytest.mark.parametrize("field", list(llm_agent.EXTRACTION_TEXT_LIMITS))
def test_native_lmfe_rejects_overlong_field_and_keeps_fast_path(field):
    schema = llm_agent.indexed_extraction_output_schema(PAYLOAD)["properties"]["candidates"]["items"]["properties"][field]
    limit = llm_agent.EXTRACTION_TEXT_LIMITS[field]
    assert "pattern" not in schema
    parser = consume(schema, '"')
    assert parser.shortcut_key()[0] == "json_freetext"
    assert consume(schema, json.dumps("x" * limit)).can_end()
    assert consume(schema, json.dumps("x" * (limit + 1))) is None
    validator = Draft202012Validator(llm_agent.indexed_extraction_output_schema(PAYLOAD))
    validator.validate({"candidates": [candidate(**{field: "x" * limit})]})
    with pytest.raises(ValidationError):
        validator.validate({"candidates": [candidate(**{field: "x" * (limit + 1)})]})
    if field != "parameter":
        assert consume(schema, 'null').can_end()
        validator.validate({"candidates": [candidate(**{field: None})]})


@pytest.mark.parametrize("text", ['l × kg(-1)', 'µg/(min × kg)', 'source says "bound"', 'path\\source', 'line\nnext'])
def test_unicode_and_escaped_text_are_not_broken_by_bounds(text):
    schema = llm_agent.indexed_extraction_output_schema(PAYLOAD)["properties"]["candidates"]["items"]["properties"]["condition"]
    assert consume(schema, json.dumps(text, ensure_ascii=False)).can_end()


def test_long_original_source_is_preserved_and_wrong_unit_still_fails_source_check():
    quote = QUOTE + " Original context." * 300
    payload = {**PAYLOAD, "input_items": [quote], "input_text_or_table": quote}
    original = json.dumps(payload, ensure_ascii=False)
    schema = llm_agent.extraction_output_schema(payload)
    assert schema["properties"]["candidates"]["items"]["properties"]["source_text"] == {"enum": [quote]}
    restored = llm_agent.restore_indexed_extraction({"candidates": [candidate()]}, payload)
    assert restored[0]["source_text"] == quote
    assert json.dumps(payload, ensure_ascii=False) == original
    # Being within the decoder length limit does not make a unit source-backed.
    wrong = "l" + " × kg(-1)" * 8
    Draft202012Validator(llm_agent.indexed_extraction_output_schema(payload)).validate(
        {"candidates": [candidate(unit=wrong)]})
    assert not _unit_appears_in_source(wrong, quote)


def test_comparison_recovery_preserves_null_mask_and_bounds_unambiguous_fields():
    comparison = "Paclitaxel half-life was 7.88 h vs 5.37 h."
    schema = llm_agent.extraction_recovery_schema({**PAYLOAD, "input_items": [comparison, QUOTE],
        "generation_recovery_mode": "comparison_schema_v1"})
    branches = schema["properties"]["candidates"]["items"]["anyOf"]
    assert branches[0]["properties"]["unit"] == {"type": "null"}
    assert branches[0]["properties"]["value_qualifier"] == {"type": "null"}
    for field, limit in llm_agent.EXTRACTION_TEXT_LIMITS.items():
        field_schema = branches[1]["properties"][field]
        string_schema = field_schema["anyOf"][0] if "anyOf" in field_schema else field_schema
        assert string_schema["maxLength"] == limit
    assert branches[1]["properties"]["source_text"] == {"const": QUOTE}


def test_actual_token_callback_blocks_merged_token_crossing_unit_limit():
    import transformers.tokenization_utils as old_location
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(old_location, "PreTrainedTokenizerBase"):
        old_location.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    tokens = [(i, chr(i), False) for i in range(128)] + [(200, "kg(-1)", False), (201, " × kg(-1)", False)]
    table = {key: text for key, text, _ in tokens}
    data = TokenEnforcerTokenizerData(tokens, lambda ids: ''.join(table.get(i, '') for i in ids), [300], False, 301)
    unit = llm_agent.indexed_extraction_output_schema(PAYLOAD)["properties"]["candidates"]["items"]["properties"]["unit"]
    schema = {"type": "object", "properties": {"unit": unit}, "required": ["unit"]}
    body = {"messages": [{"role": "user", "content": QUOTE}],
            "response_format": {"json_schema": {"schema": schema}}}
    enforcer = server.evidence_json_callback(server.evidence_tokenizer_data(data), body).token_enforcer
    prefix = [999]
    for character in '{"unit":"' + 'x' * 127:
        assert ord(character) in enforcer.get_allowed_tokens(prefix).allowed_tokens
        prefix.append(ord(character))
    allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
    assert 200 not in allowed and 201 not in allowed
    assert ord('x') in allowed and ord('"') in allowed
    prefix.append(ord('x'))
    allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
    assert ord('x') not in allowed and ord('"') in allowed
    assert 300 not in allowed
