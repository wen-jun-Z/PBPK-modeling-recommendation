"""Real LMFE token callbacks; byte-fallback Unicode absent from the prompt."""
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from pbpk_kb.processing import evidence_json_constraints as constraints

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("unicode_evidence_server", ROOT / "scripts/deployment/evidence_qwen_server.py")
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)

UNICODE = ["⁻", "∗", "é", "药", "𠀀", "🧪", "𝑨", "\u2028", "\u2029", "\uffff", "\U0010ffff"]


def callback_for(character, value_schema):
    import transformers.tokenization_utils as old_location
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(old_location, "PreTrainedTokenizerBase"):
        old_location.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    tokens = [(i, chr(i), False) for i in range(128)]
    tokens += [(201, "�", False), (202, "�", False), (203, "\ud800", False), (204, "\udfff", False)]
    table = {i: text for i, text, _ in tokens}
    def decode(ids):
        return "".join(table.get(i, "") for i in ids).replace("��", character).rstrip("�")
    data = TokenEnforcerTokenizerData(tokens, decode, [300], False, 301)
    schema = {"type": "object", "properties": {"v": value_schema}, "required": ["v"], "additionalProperties": False}
    body = {"messages": [{"role": "user", "content": "Source uses ng h ml minus sign1; return JSON."}],
            "response_format": {"json_schema": {"schema": schema}}}
    assert character not in json.dumps(body, ensure_ascii=False)
    enforcer = server.evidence_json_callback(server.evidence_tokenizer_data(data), body).token_enforcer
    assert character not in enforcer.root_parser.config.alphabet
    return enforcer, schema, decode


def feed(enforcer, prefix, tokens):
    for token in tokens:
        allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
        assert 300 not in allowed, "EOS must not be enabled before the JSON root closes"
        assert token in allowed
        prefix.append(token)


@pytest.mark.parametrize("character", UNICODE)
@pytest.mark.parametrize("nullable", [False, True])
def test_unknown_unicode_preserves_string_stack_and_native_length_boundary(character, nullable):
    value_schema = {"type": "string", "minLength": 1, "maxLength": 2}
    if nullable:
        value_schema = {"anyOf": [value_schema, {"type": "null"}]}
    enforcer, schema, decode = callback_for(character, value_schema)
    alphabet = enforcer.root_parser.config.alphabet
    prefix = [999]
    feed(enforcer, prefix, [ord(c) for c in '{"v":"a'])
    state = enforcer.prefix_states[tuple(prefix[:-1])].parser
    assert state.shortcut_key()[0] == "json_freetext"
    feed(enforcer, prefix, [201, 202])
    allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
    assert 300 not in allowed
    assert ord("x") not in allowed
    assert 201 not in allowed and 202 not in allowed
    assert len(enforcer.prefix_states[tuple(prefix)].parser.object_stack) > 0
    feed(enforcer, prefix, [ord('"'), ord('}')])
    assert 300 in enforcer.get_allowed_tokens(prefix).allowed_tokens
    parsed = json.loads(decode(prefix[1:]))
    assert parsed == {"v": "a" + character}
    Draft202012Validator(schema).validate(parsed)
    assert enforcer.root_parser.config.alphabet == alphabet
    assert constraints._CONSUMED_UNICODE.get() is None


@pytest.mark.parametrize("value_schema", [
    {"enum": ["safe"]}, {"const": "safe"}, {"type": "string", "pattern": "[a-z]+"},
])
def test_unicode_does_not_bypass_enum_const_or_pattern(value_schema):
    enforcer, _, _ = callback_for("⁻", value_schema)
    prefix = [999]
    feed(enforcer, prefix, [ord(c) for c in '{"v":"'])
    allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
    assert 201 not in allowed and 202 not in allowed and 300 not in allowed
    feed(enforcer, prefix, [ord(c) for c in 'safe"}'])
    assert 300 in enforcer.get_allowed_tokens(prefix).allowed_tokens


def test_controls_and_raw_surrogates_blocked_but_escaped_quotes_remain_valid():
    enforcer, schema, decode = callback_for("⁻", {"type": "string", "maxLength": 128})
    prefix = [999]
    feed(enforcer, prefix, [ord(c) for c in '{"v":"'])
    allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
    assert all(i not in allowed for i in range(32))
    assert 203 not in allowed and 204 not in allowed
    feed(enforcer, prefix, [201, 202])
    feed(enforcer, prefix, [ord(c) for c in r'\"quoted\"\n\t\\path"}'])
    assert 300 in enforcer.get_allowed_tokens(prefix).allowed_tokens
    parsed = json.loads(decode(prefix[1:]))
    assert parsed == {"v": '⁻"quoted"\n\t\\path'}
    Draft202012Validator(schema).validate(parsed)


def test_unicode_context_and_alphabets_remain_isolated_between_concurrent_callbacks():
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(test_unknown_unicode_preserves_string_stack_and_native_length_boundary, c, True)
                   for c in ("⁻", "🧪")]
        for future in futures:
            future.result()
    assert constraints._CONSUMED_UNICODE.get() is None


def test_patch_installation_is_idempotent_and_records_loaded_source():
    import hashlib
    from lmformatenforcer import jsonschemaparser
    constraints.install_unicode_freetext_consumption()
    installed = jsonschemaparser.JsonSchemaParser.add_character
    constraints.install_unicode_freetext_consumption()
    assert jsonschemaparser.JsonSchemaParser.add_character is installed
    assert constraints.unicode_freetext_compatibility() == {
        "schema_unicode_compatibility": "unicode_freetext_consumption_v2",
        "schema_unicode_patch_sha256": hashlib.sha256(Path(constraints.__file__).read_bytes()).hexdigest(),
    }
