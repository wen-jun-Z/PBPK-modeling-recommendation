"""Exercise field order through the production LMFE token callback."""
import importlib.util
from pathlib import Path

import pytest

from pbpk_kb.processing.llm_agent import indexed_extraction_output_schema


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "indexed_order_evidence_server", ROOT / "scripts/deployment/evidence_qwen_server.py",
)
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def _enforcer(schema):
    import transformers.tokenization_utils as old_location
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData

    if not hasattr(old_location, "PreTrainedTokenizerBase"):
        old_location.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    tokens = [(value, chr(value), False) for value in range(128)]
    data = TokenEnforcerTokenizerData(
        tokens, lambda values: "".join(chr(value) for value in values), [128], False, 129,
    )
    body = {
        "messages": [{"role": "user", "content": "Amoxicillin half-life was 1 h."}],
        "response_format": {"json_schema": {"schema": schema}},
    }
    server.install_json_const_compatibility()
    return server.evidence_json_callback(data, body).token_enforcer


def _allowed_after(enforcer, prefix):
    tokens = [999]
    for char in prefix:
        assert ord(char) in enforcer.get_allowed_tokens(tokens).allowed_tokens
        tokens.append(ord(char))
    return enforcer.get_allowed_tokens(tokens).allowed_tokens


def _schema():
    return indexed_extraction_output_schema({
        "target_compound": "amoxicillin", "input_items": ["Amoxicillin half-life was 1 h."],
    })


def test_indexed_extraction_must_assess_then_reason_before_candidates():
    enforcer = _enforcer(_schema())
    assert enforcer.root_parser.config.force_json_field_order
    allowed = _allowed_after(enforcer, '{"')
    assert ord("a") in allowed
    assert ord("c") not in allowed and ord("r") not in allowed
    allowed = _allowed_after(enforcer, '{"assessment":"extract","')
    assert ord("r") in allowed and ord("c") not in allowed
    allowed = _allowed_after(enforcer, '{"assessment":"extract","reason":"Source fact.","')
    assert ord("c") in allowed


@pytest.mark.parametrize("decision,empty", [("extract", False), ("no_evidence", True), ("uncertain", True)])
def test_native_array_requires_extraction_and_preserves_legitimate_abstention(decision, empty):
    enforcer = _enforcer(_schema())
    prefix = '{"assessment":"' + decision + '","reason":"Source decision.","candidates":['
    allowed = _allowed_after(enforcer, prefix)
    assert (ord("]") in allowed) is empty
    if empty:
        assert 128 in _allowed_after(enforcer, prefix + "]}")
    else:
        assert ord("{") in allowed


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
     "required": ["supported", "reason"], "additionalProperties": False},
    {"type": "object", "properties": {"assessment": {"enum": ["yes", "no"]},
      "reason": {"type": "string"}, "candidates": {"type": "array", "items": {"type": "string"}}},
     "required": ["assessment", "reason", "candidates"], "additionalProperties": False},
])
def test_other_response_schemas_keep_their_existing_field_order_policy(schema):
    enforcer = _enforcer(schema)
    assert not enforcer.root_parser.config.force_json_field_order
    assert ord("r") in _allowed_after(enforcer, '{"')
