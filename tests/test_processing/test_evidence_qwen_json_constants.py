"""Exercise the actual LMFE character parser, not only schema metadata."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("evidence_qwen_server", ROOT / "scripts/deployment/evidence_qwen_server.py")
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def accepts(schema, text):
    from lmformatenforcer import JsonSchemaParser
    server.install_json_const_compatibility()
    parser = JsonSchemaParser(schema)
    for char in text:
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


@pytest.mark.parametrize("value,other", [(True, False), (False, True), (None, 0), (0, None), (2.5, 2.6), (-3, 3)])
def test_exact_scalar_const_remains_exact(value, other):
    schema = {"type": "object", "properties": {"value": {"const": value}},
              "required": ["value"], "additionalProperties": False}
    original = json.dumps(schema, sort_keys=True)
    assert accepts(schema, json.dumps({"value": value}))
    assert not accepts(schema, json.dumps({"value": other}))
    assert json.dumps(schema, sort_keys=True) == original


def test_real_extraction_schema_requires_review_true():
    from pbpk_kb.processing.llm_agent import EXTRACTION_OUTPUT_SCHEMA
    candidate = {"target_compound": "caffeine", "evidence_type": "clinical_pk",
                 "parameter": "Cmax", "value": 2.5, "value_low": None, "value_high": None,
                 "value_qualifier": None, "unit": "mg/L", "condition": None,
                 "source_text": "Caffeine Cmax was 2.5 mg/L.", "confidence": .8, "requires_review": True}
    assert accepts(EXTRACTION_OUTPUT_SCHEMA, json.dumps({"candidates": [candidate]}))
    candidate["requires_review"] = False
    assert not accepts(EXTRACTION_OUTPUT_SCHEMA, json.dumps({"candidates": [candidate]}))


def test_actual_decoder_rejects_joined_or_rewritten_source_quotes():
    from pbpk_kb.processing.llm_agent import extraction_output_schema
    quote = 'Bedaquiline is an in vitro substrate of CYP3A4.'
    other = 'Bedaquiline protein binding was measured.'
    schema = extraction_output_schema({'target_compound': 'bedaquiline', 'input_items': [quote, other]})
    candidate = {'target_compound': 'bedaquiline', 'evidence_type': 'adme', 'parameter': 'CYP3A4',
                 'value': None, 'value_low': None, 'value_high': None, 'value_qualifier': None,
                 'unit': None, 'condition': None, 'source_text': quote, 'confidence': .8, 'requires_review': True}
    assert accepts(schema, json.dumps({'candidates': [candidate]}))
    for text in [quote + '\n' + other, quote + '\n', quote.lower()]:
        assert not accepts(schema, json.dumps({'candidates': [{**candidate, 'source_text': text}]}))


def byte_fallback_tokenizer_data():
    """Small tokenizer with the real Qwen failure shape: two bytes form U+2217."""
    # Production ComponentGenerator initializes this LMFE/Transformers 5 alias
    # through prepare_json_constraint before constructing any callback.
    import transformers.tokenization_utils as old_location
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(old_location, 'PreTrainedTokenizerBase'):
        old_location.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
    tokens = [(value, chr(value), False) for value in range(128)]
    tokens += [(200, 'μ', False), (201, '�', False), (202, '�', False)]
    table = {token: text for token, text, _ in tokens}

    def decode(ids):
        return ''.join(table.get(token, '') for token in ids).replace('��', '∗').rstrip('�')

    return TokenEnforcerTokenizerData(tokens, decode, [300], False, 301)


def unit_body():
    return {'messages': [{'role': 'user', 'content': 'AUC was 958μg∗h/L.'}],
            'response_format': {'json_schema': {'schema': {
                'type': 'object', 'properties': {'unit': {'type': ['string', 'null']}},
                'required': ['unit'], 'additionalProperties': False}}}}


def test_byte_fallback_source_unicode_never_ends_json_inside_unit():
    original = byte_fallback_tokenizer_data()
    assert 'μ' in original.tokenizer_alphabet and '∗' not in original.tokenizer_alphabet
    prepared = server.evidence_tokenizer_data(original)
    callback = server.evidence_json_callback(prepared, unit_body())
    enforcer = callback.token_enforcer
    tokens = [ord(char) for char in '{"unit":"'] + [200, ord('g'), 201, 202]
    tokens += [ord(char) for char in 'h/L"}']
    prefix = [999]  # Separate prompt: first call initializes the output parser.
    for token in tokens:
        allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
        assert token in allowed
        assert 300 not in allowed
        prefix.append(token)
    assert 300 in enforcer.get_allowed_tokens(prefix).allowed_tokens
    assert json.loads(original.decoder(tokens))['unit'] == 'μg∗h/L'
    assert '∗' not in original.tokenizer_alphabet  # Per-request isolation.
    assert enforcer.root_parser.config.max_consecutive_whitespaces == 2
    assert enforcer.root_parser.config.max_json_array_length == 512


def test_json_free_text_rejects_raw_controls_but_keeps_escaped_controls():
    data = server.evidence_tokenizer_data(byte_fallback_tokenizer_data())
    enforcer = server.evidence_json_callback(data, unit_body()).token_enforcer
    prefix = [999]
    for char in '{"unit":"':
        assert ord(char) in enforcer.get_allowed_tokens(prefix).allowed_tokens
        prefix.append(ord(char))
    allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
    assert all(value not in allowed for value in range(32))
    assert 300 not in allowed
    for char in '\\t"}':
        assert ord(char) in enforcer.get_allowed_tokens(prefix).allowed_tokens
        prefix.append(ord(char))
    assert 300 in enforcer.get_allowed_tokens(prefix).allowed_tokens
