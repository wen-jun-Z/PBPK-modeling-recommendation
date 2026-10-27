"""Grade-first JSON with citations required for every known decision.

This prospective decoder preserves i,g,a,e order. The model chooses its grade
and applicability before it chooses citations; no supporting ID is inserted.
The citation requirement enforces the existing answer contract, not entailment.
Old prompts, decoders, model settings and experimental records are unchanged.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
import frontier_evidence_first_decoding as shared

VERSION = "pbpk-grade-first-conditional-citation-json-v1"
FIELD_ORDER = ("i", "g", "a", "e")
MAX_WHITESPACE = shared.MAX_WHITESPACE
_OPEN_EVIDENCE = re.compile(
    r'"g"\s*:\s*(null|[0-3])\s*,\s*"a"\s*:\s*"([YNU])"\s*,\s*"e"\s*:\s*\[([^\]]*)$')


def conditional_citation_schema(request_count: int, reference_count: int) -> dict:
    """Base syntax; the parser supplies the conditional citation minimum.

    Conditional JSONSchema keywords are deliberately not passed to LMFE.
    Empty documents constrain grade/applicability before citation generation,
    avoiding prefixes that could never satisfy the existing answer contract.
    """
    schema = shared.evidence_first_schema(request_count, reference_count)
    item = schema["items"]
    item["required"] = list(FIELD_ORDER)
    item["properties"] = {key: item["properties"][key] for key in FIELD_ORDER}
    if reference_count == 0:
        item["properties"]["g"] = {"type": "null"}
        item["properties"]["a"] = {"type": "string", "enum": ["U"]}
    return schema


class ConditionalCitationParser(shared.EvidenceFirstParser):
    """Immutable syntax intersection; only the reference-ID helper is reused."""

    def get_allowed_characters(self):
        allowed = self.inner.get_allowed_characters().replace("\\", "")
        match = _OPEN_EVIDENCE.search(self.prefix)
        if match:
            grade, applicability, content = match.groups()
            allowed = self._evidence_characters(content, allowed)
            if (grade != "null" or applicability != "U") and not content.strip():
                allowed = allowed.replace("]", "")
        return allowed

    def add_character(self, character):
        if len(character) != 1 or character not in self.get_allowed_characters():
            raise ValueError("Character violates conditional-citation JSON constraints")
        return ConditionalCitationParser(self.inner.add_character(character),
                                          self.reference_count, self.prefix + character)


def conditional_citation_parser(request_count: int, reference_count: int) -> ConditionalCitationParser:
    schema = conditional_citation_schema(request_count, reference_count)
    # Retain the frozen workaround for LMFE's numeric-enum array counter:
    # integer syntax inside; exact range, uniqueness and <=3 in the intersection.
    schema["items"]["properties"]["e"]["items"] = {"type": "integer"}
    inner = JsonSchemaParser(schema, config=CharacterLevelParserConfig(
        max_consecutive_whitespaces=MAX_WHITESPACE, force_json_field_order=True,
        max_json_array_length=max(3, request_count)))
    return ConditionalCitationParser(inner, reference_count)


class ConditionalCitationConstraint:
    """Reuse official tokenizer/EOS identity; isolate each row and generation."""

    def __init__(self, base):
        self.base = base
        self.tokenizer_data = base.tokenizer_data
        self.eos_token_ids = base.eos_token_ids

    def for_tasks(self, tasks):
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

        if not tasks:
            raise ValueError("Cannot constrain an empty generation batch")
        callbacks = []
        for packet, queries in tasks:
            if not packet["refs"] and packet.get("body", {}).get("excerpts"):
                raise ValueError("Empty refs disagree with supplied excerpts")
            parser = conditional_citation_parser(len(queries), len(packet["refs"]))
            callbacks.append(build_transformers_prefix_allowed_tokens_fn(self.tokenizer_data, parser))

        def allowed_tokens(batch_id, token_ids):
            if type(batch_id) is not int or not 0 <= batch_id < len(callbacks):
                raise ValueError("Unknown conditional-citation generation batch row")
            return callbacks[batch_id](0, token_ids)

        return allowed_tokens


def conditional_citation_constraint(base):
    return base if isinstance(base, ConditionalCitationConstraint) else ConditionalCitationConstraint(base)


def decoding_protocol():
    schema = conditional_citation_schema(1, 1)
    return {"version": VERSION,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_implementation_sha256": {
            Path(shared.__file__).name: hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest()},
        "library": "lm-format-enforcer", "library_version": version("lm-format-enforcer"),
        "schema_template": schema,
        "schema_template_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "field_order": list(FIELD_ORDER), "canonical_ascii_keys_and_values": True,
        "decision_order": "Model chooses g and a before choosing supplied evidence IDs; no score or citation is inserted",
        "conditional_minimum": "g != null OR a != U requires 1..3 unique supplied references; g=null AND a=U permits 0..3",
        "reference_domain": "1..supplied_excerpt_count; <=3 unique IDs, chosen by the model",
        "uniqueness_enforcement": "frozen immutable reference numeric-prefix helper; not JSONSchema uniqueItems alone",
        "lmfe_numeric_enum_counter_workaround": "inner generic integer syntax; exact reference domain and <=3 enforced by character intersection",
        "nonempty_evidence": {"grades": [None, 0, 1, 2, 3], "applicability": ["Y", "N", "U"], "combinations": 15},
        "empty_evidence": {"grade": None, "applicability": "U"},
        "empty_document": "Only g=null,a=U,e=[] is structurally allowed; a model call is unnecessary in production",
        "answer_space": "Every answer object accepted by original parse_answers remains representable in i,g,a,e order",
        "max_consecutive_whitespaces": MAX_WHITESPACE,
        "eos_policy": "reuse all original official model EOS IDs unchanged",
        "state_policy": "fresh character parser and token enforcer per batch row and generation call; tokenizer index reused",
        "semantic_entailment_verified": False, "citations_inserted": False,
        "scores_used": False, "posthoc_answer_fixes": False,
        "request_id_uniqueness_checked_after_decoding": True,
        "empty_evidence_abstention_is_enforced_not_measured_model_ability": True,
        "known_answer_citation_presence_is_enforced_not_measured_entailment": True,
        "token_limit_may_truncate": True, "existing_answer_and_scope_validation_still_required": True,
        "prospective_control_and_source_validation_required_before_activation": True}
