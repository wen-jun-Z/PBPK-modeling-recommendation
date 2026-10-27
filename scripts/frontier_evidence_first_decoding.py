"""Evidence-first JSON decoding with the existing valid-answer space.

The model selects supplied citation numbers before choosing grade/applicability.
An empty citation list admits only null/U, exactly as the existing post-decoding
validator requires. Nonempty citations retain all 15 grade/applicability choices.
This syntax/consistency constraint neither finds evidence nor verifies entailment.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

from lmformatenforcer import CharacterLevelParser, CharacterLevelParserConfig, JsonSchemaParser

VERSION = "pbpk-evidence-first-json-v1"
FIELD_ORDER = ("i", "e", "g", "a")
MAX_WHITESPACE = 2
_WHITESPACE = " \t\r\n"
_OPEN_EVIDENCE = re.compile(r'"e"\s*:\s*\[([^\]]*)$')
_EMPTY_GRADE = re.compile(r'"e"\s*:\s*\[\s*\]\s*,\s*"g"\s*:\s*([^,}]*)$')
_EMPTY_APPLICABILITY = re.compile(r'"e"\s*:\s*\[\s*\]\s*,\s*"g"\s*:\s*null\s*,\s*"a"\s*:\s*"([^"\\]*)$')


def evidence_first_schema(request_count: int, reference_count: int) -> dict:
    """Base JSON syntax; the parser also enforces unique refs and empty => null/U.

    For zero refs, the unreachable item placeholder avoids LMFE's unsupported
    empty enum. Both maxItems=0 and the character intersection forbid any item.
    No placeholder reference is supplied to the model or can be generated.
    """
    if type(request_count) is not int or request_count < 1:
        raise ValueError("request_count must be a positive integer")
    if type(reference_count) is not int or reference_count < 0:
        raise ValueError("reference_count must be a nonnegative integer")
    return {"type": "array", "minItems": request_count, "maxItems": request_count,
        "items": {"type": "object", "additionalProperties": False,
            "required": list(FIELD_ORDER), "properties": {
                "i": {"type": "integer", "enum": list(range(1, request_count + 1))},
                "e": {"type": "array", "minItems": 0, "maxItems": min(3, reference_count),
                      "uniqueItems": True,
                      "items": {"type": "integer", "enum": list(range(1, max(1, reference_count) + 1))}},
                "g": {"anyOf": [{"type": "integer", "enum": [0, 1, 2, 3]}, {"type": "null"}]},
                "a": {"type": "string", "enum": ["Y", "N", "U"]},
            }}}


class EvidenceFirstParser(CharacterLevelParser):
    """Immutable intersection around JSONSchema; no unsupported if/then logic.

    Literal keys and values are ASCII, so escaped spellings are unnecessary and
    are excluded. With fixed field order there is no free text in which the
    anchored current-field patterns could accidentally match another value.
    Numeric lookahead preserves distinct multi-digit refs sharing a prefix.
    """

    def __init__(self, inner, reference_count, prefix=""):
        super().__init__(inner.config)
        self.inner = inner
        self.reference_count = reference_count
        self.prefix = prefix

    def _evidence_characters(self, content, allowed):
        segments = content.split(",")
        used = tuple(int(value.strip()) for value in segments[:-1])
        fragment = segments[-1].strip()
        remaining = tuple(str(n) for n in range(1, self.reference_count + 1) if n not in used)
        maximum = min(3, self.reference_count)
        result = []
        for character in allowed:
            if character in _WHITESPACE:
                result.append(character)
            elif character.isascii() and character.isdigit():
                if len(used) < maximum and any(n.startswith(fragment + character) for n in remaining):
                    result.append(character)
            elif character == "]":
                if (not used and not fragment and len(segments) == 1) or fragment in remaining:
                    result.append(character)
            elif character == "," and fragment in remaining and len(used) + 1 < maximum:
                result.append(character)
        return "".join(result)

    def get_allowed_characters(self):
        allowed = self.inner.get_allowed_characters().replace("\\", "")
        match = _OPEN_EVIDENCE.search(self.prefix)
        if match:
            return self._evidence_characters(match.group(1), allowed)
        match = _EMPTY_GRADE.search(self.prefix)
        if match:
            fragment = match.group(1).strip()
            if len(fragment) < 4:
                next_char = "null"[len(fragment)]
                return "".join(c for c in allowed if c in _WHITESPACE or c == next_char)
        match = _EMPTY_APPLICABILITY.search(self.prefix)
        if match:
            fragment = match.group(1)
            return "".join(c for c in allowed if c == ("U" if not fragment else '"'))
        return allowed

    def add_character(self, character):
        if len(character) != 1 or character not in self.get_allowed_characters():
            raise ValueError("Character violates evidence-first JSON constraints")
        return EvidenceFirstParser(self.inner.add_character(character), self.reference_count,
                                   self.prefix + character)

    def can_end(self):
        return self.inner.can_end()


def evidence_first_parser(request_count: int, reference_count: int) -> EvidenceFirstParser:
    schema = evidence_first_schema(request_count, reference_count)
    # LMFE 0.11.3 may double-count completed numeric enum items and reject a
    # valid third citation at maxItems=3. Use its generic integer syntax here;
    # the immutable intersection supplies the exact enum/uniqueness/count rules.
    schema["items"]["properties"]["e"]["items"] = {"type": "integer"}
    inner = JsonSchemaParser(schema,
        config=CharacterLevelParserConfig(max_consecutive_whitespaces=MAX_WHITESPACE,
            force_json_field_order=True, max_json_array_length=max(3, request_count)))
    return EvidenceFirstParser(inner, reference_count)


class EvidenceFirstConstraint:
    """Reuse the original tokenizer/EOS; isolate every batch row and call."""

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
            parser = evidence_first_parser(len(queries), len(packet["refs"]))
            callbacks.append(build_transformers_prefix_allowed_tokens_fn(self.tokenizer_data, parser))

        def allowed_tokens(batch_id, token_ids):
            if type(batch_id) is not int or not 0 <= batch_id < len(callbacks):
                raise ValueError("Unknown evidence-first generation batch row")
            return callbacks[batch_id](0, token_ids)

        return allowed_tokens


def evidence_first_constraint(base):
    return base if isinstance(base, EvidenceFirstConstraint) else EvidenceFirstConstraint(base)


def decoding_protocol():
    schema = evidence_first_schema(1, 1)
    return {"version": VERSION, "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "library": "lm-format-enforcer", "library_version": version("lm-format-enforcer"),
            "schema_template": schema,
            "schema_template_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "field_order": list(FIELD_ORDER), "canonical_ascii_keys_and_values": True,
            "reference_domain": "1..supplied_excerpt_count; choose zero to three unique supplied IDs",
            "uniqueness_enforcement": "immutable character-parser numeric-prefix lookahead; not JSONSchema uniqueItems alone",
            "lmfe_numeric_enum_counter_workaround": "inner citation item uses generic integer syntax; character intersection enforces exact supplied-reference domain and max three",
            "empty_evidence": {"grade": None, "applicability": "U"},
            "nonempty_evidence": {"grades": [None, 0, 1, 2, 3], "applicability": ["Y", "N", "U"], "combinations": 15},
            "empty_document": "Only e=[],g=null,a=U is syntactically valid; a model call is unnecessary in production",
            "answer_space": "Every answer object accepted by the original parse_answers remains representable in i,e,g,a order",
            "max_consecutive_whitespaces": MAX_WHITESPACE,
            "eos_policy": "reuse all original official model EOS IDs unchanged",
            "state_policy": "fresh character parser and token enforcer per batch row and generation call; tokenizer index reused",
            "semantic_entailment_verified": False, "citations_inserted": False,
            "scores_used": False, "posthoc_answer_fixes": False,
            "request_id_uniqueness_checked_after_decoding": True,
            "empty_evidence_abstention_is_enforced_not_measured_model_ability": True,
            "token_limit_may_truncate": True, "existing_answer_and_scope_validation_still_required": True}
