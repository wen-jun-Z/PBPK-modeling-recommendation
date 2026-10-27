"""Actual project LMFE character/token contracts; no model or GPU is loaded."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from pbpk_kb.processing import evidence_json_constraints as constraints
from pbpk_kb.processing.auto_quality_llm_stage import REPAIR_PROMPT, quality_output_schema


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("strict_evidence_qwen_server", ROOT / "scripts/deployment/evidence_qwen_server.py")
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def parser_after(schema, text):
    from lmformatenforcer import JsonSchemaParser
    constraints.install_strict_object_separators()
    parser = JsonSchemaParser(schema)
    for character in text:
        if character not in parser.get_allowed_characters():
            return None
        parser = parser.add_character(character)
    return parser


def accepts(schema, text):
    parser = parser_after(schema, text)
    return parser is not None and parser.can_end()


def test_real_quality_repair_rejects_trailing_comma_without_changing_schema():
    schema = quality_output_schema(REPAIR_PROMPT, {"evidence_id": "e1"})
    before = json.dumps(schema, sort_keys=True)
    response = '{"evidence_id":"e1","repairs":{"value_qualifier":"high", },"supporting_text":"high molecular weight","confidence":0.95}'
    assert not accepts(schema, response)
    assert accepts(schema, response.replace('"high", }', '"high"}'))
    empty = '{"evidence_id":"e1","repairs":{},"supporting_text":"No number reported.","confidence":0.9}'
    assert accepts(schema, empty)
    assert json.dumps(schema, sort_keys=True) == before


@pytest.mark.parametrize("value,value_schema", [
    ('"x"', {"type": "string"}), ('1', {"type": "number"}),
    ('null', {"type": "null"}), ('true', {"type": "boolean"}),
    ('{}', {"type": "object", "properties": {}}),
    ('[]', {"type": "array", "items": {"type": "number"}}),
])
@pytest.mark.parametrize("space", ["", " ", "\n "])
def test_required_keys_met_does_not_allow_optional_key_separator_to_close(value, value_schema, space):
    schema = {"type": "object", "required": ["r"], "properties": {
        "r": value_schema, "optional": {"type": "string"}}, "additionalProperties": False}
    prefix = '{"r":' + value + ',' + space
    parser = parser_after(schema, prefix)
    assert parser is not None
    assert "}" not in parser.get_allowed_characters()
    assert '"' in parser.get_allowed_characters()
    assert accepts(schema, '{"r":' + value + '}')
    assert accepts(schema, prefix + '"optional":"ok"}')
    assert not accepts(schema, prefix + '}')


def test_nested_objects_and_array_items_keep_independent_separator_states():
    inner = {"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "string"}}}
    schema = {"type": "object", "properties": {
        "child": inner, "items": {"type": "array", "items": inner}}}
    for valid in ['{}', '{"child":{}}', '{"child":{"x":"ok"},"items":[{}, {"x":"x","y":"y"}]}']:
        assert accepts(schema, valid)
    for invalid in ['{"child":{"x":"ok",}}', '{"child":{},}', '{"items":[{"x":"x",}]}', '{"items":[{},]}']:
        assert not accepts(schema, invalid)


@pytest.mark.parametrize("item_schema,value", [
    ({"type": "string"}, '"x"'), ({"type": "number"}, '1'),
    ({"type": "object", "properties": {}}, '{}'), ({"type": "array", "items": {"type": "number"}}, '[1]'),
])
def test_existing_array_separator_constraint_stays_strict(item_schema, value):
    schema = {"type": "array", "items": item_schema}
    assert accepts(schema, '[]')
    assert accepts(schema, '[' + value + ']')
    assert accepts(schema, '[' + value + ',' + value + ']')
    assert not accepts(schema, '[' + value + ', ]')


def test_patch_preserves_escaped_quotes_native_length_limit_and_fast_path():
    schema = {"type": "object", "properties": {"quote": {"type": "string", "maxLength": 512}}}
    quote = 'High "molecular weight"\npath\\source.'
    assert accepts(schema, json.dumps({"quote": quote}))
    parser = parser_after(schema, '{"quote":"')
    assert parser.shortcut_key()[0] == "json_freetext"
    assert accepts(schema, json.dumps({"quote": "x" * 512}))
    assert not accepts(schema, json.dumps({"quote": "x" * 513}))


def test_actual_token_callback_blocks_merged_trailing_comma_tokens():
    import transformers.tokenization_utils as old_location
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(old_location, "PreTrainedTokenizerBase"):
        old_location.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    tokens = [(i, chr(i), False) for i in range(128)] + [(200, ',}', False), (201, ',\n }', False)]
    table = {key: text for key, text, _ in tokens}
    data = TokenEnforcerTokenizerData(tokens, lambda ids: ''.join(table.get(i, '') for i in ids), [300], False, 301)
    schema = {"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "string"}}}
    body = {"messages": [{"role": "user", "content": 'Source says "x".'}],
            "response_format": {"json_schema": {"schema": schema}}}
    enforcer = server.evidence_json_callback(server.evidence_tokenizer_data(data), body).token_enforcer
    prefix = [999]
    for character in '{"x":"x"':
        assert ord(character) in enforcer.get_allowed_tokens(prefix).allowed_tokens
        prefix.append(ord(character))
    allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
    assert 200 not in allowed and 201 not in allowed
    assert ord('}') in allowed and ord(',') in allowed
    prefix.append(ord(','))
    allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
    assert ord('}') not in allowed and 300 not in allowed
    assert ord('"') in allowed


def test_installer_is_idempotent_and_status_records_loaded_patch(monkeypatch):
    from lmformatenforcer import jsonschemaparser
    constraints.install_strict_object_separators()
    original = jsonschemaparser.ObjectParsingState.get_allowed_characters
    constraints.install_strict_object_separators()
    assert jsonschemaparser.ObjectParsingState.get_allowed_characters is original
    writes = []
    monkeypatch.setattr(server, "write_json", lambda path, value: writes.append(dict(value)))
    monkeypatch.setattr(server.threading.Thread, "start", lambda self: None)
    engine = server.Engine()
    assert engine.state["schema_object_separator_compatibility"] == constraints.OBJECT_SEPARATOR_COMPATIBILITY
    assert engine.state["schema_object_separator_patch_sha256"] == hashlib.sha256(Path(constraints.__file__).read_bytes()).hexdigest()
    assert writes[-1]["schema_object_separator_patch_sha256"] == engine.state["schema_object_separator_patch_sha256"]
    assert engine.state["schema_unicode_compatibility"] == constraints.UNICODE_FREETEXT_COMPATIBILITY
    assert engine.state["schema_unicode_patch_sha256"] == hashlib.sha256(Path(constraints.__file__).read_bytes()).hexdigest()
