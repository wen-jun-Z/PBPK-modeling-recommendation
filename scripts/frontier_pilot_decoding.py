"""Pilot-only empty-document grammar; no validation cases or labels are changed.

Production normally bypasses generation for empty documents. Controlled pilots
must still query the model: all grades and applicability values stay possible,
but no source citation may be invented. LMFE 0.11.3 does not enforce maxItems=0
at array opening, so an explicit immutable character parser enforces e=[].
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from lmformatenforcer import CharacterLevelParser, CharacterLevelParserConfig, JsonSchemaParser

from frontier_judge_decoding import MAX_CONSECUTIVE_WHITESPACES, judge_json_schema

VERSION = "pbpk-pilot-empty-source-decoding-v1"
_OPEN_EMPTY_EVIDENCE = re.compile(r'"e"\s*:\s*\[\s*$')


class EmptyEvidenceParser(CharacterLevelParser):
    """Intersect the regular judge grammar with an empty e-array constraint.

    The schema permits only the literal keys i/g/a/e and fixed applicability
    strings. Thus the anchored literal-key pattern cannot match within a free
    text value. JSON syntax, whitespace limits and field-order freedom continue
    to come from the original parser. No relevance/applicability value is fixed.
    """

    def __init__(self, inner, prefix=""):
        super().__init__(inner.config)
        self.inner = inner
        self.prefix = prefix

    def get_allowed_characters(self):
        allowed = self.inner.get_allowed_characters()
        if _OPEN_EMPTY_EVIDENCE.search(self.prefix):
            return "".join(c for c in allowed if c in "] \t\r\n")
        return allowed

    def add_character(self, character):
        if character not in self.get_allowed_characters():
            raise ValueError("Character violates pilot empty-evidence grammar")
        return EmptyEvidenceParser(self.inner.add_character(character), self.prefix + character)

    def can_end(self):
        return self.inner.can_end()


def empty_source_parser(request_count):
    # The one-reference base schema is never shown to the model and its e item
    # domain is unreachable under EmptyEvidenceParser. It preserves every other
    # production grammar rule, including all 15 independent g/a combinations.
    schema = judge_json_schema(request_count, 1)
    schema["items"]["properties"]["e"]["maxItems"] = 0
    inner = JsonSchemaParser(schema, config=CharacterLevelParserConfig(
        max_consecutive_whitespaces=MAX_CONSECUTIVE_WHITESPACES,
        force_json_field_order=False,
        max_json_array_length=max(3, request_count),
    ))
    return EmptyEvidenceParser(inner)


class PilotJSONConstraint:
    """Wrap an already indexed JudgeJSONConstraint; reuse official EOS IDs."""

    def __init__(self, base):
        self.base = base

    def for_tasks(self, tasks):
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

        if not tasks:
            raise ValueError("Cannot constrain an empty generation batch")
        if all(packet["refs"] for packet, _ in tasks):
            return self.base.for_tasks(tasks)
        callbacks = []
        for packet, queries in tasks:
            if packet["refs"]:
                callback = self.base.for_tasks([(packet, queries)])
            else:
                if packet.get("body", {}).get("excerpts"):
                    raise ValueError("Empty refs disagree with supplied excerpts")
                callback = build_transformers_prefix_allowed_tokens_fn(
                    self.base.tokenizer_data, empty_source_parser(len(queries)))
            callbacks.append(callback)

        def allowed_tokens(batch_id, token_ids):
            if type(batch_id) is not int or not 0 <= batch_id < len(callbacks):
                raise ValueError("Unknown pilot generation batch row")
            # Each callback is an independent one-row parser. Identical padded
            # prompts must never share empty/nonempty evidence parser state.
            return callbacks[batch_id](0, token_ids)

        return allowed_tokens


def empty_source_constraint(base):
    return PilotJSONConstraint(base)


def pilot_decoding_protocol():
    return {"version": VERSION,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "applies_to": "pilot packets with zero supplied source excerpts only",
            "empty_evidence_enforcement": "immutable character parser forbids any e-array item; maxItems=0 alone is insufficient in LMFE 0.11.3",
            "grade_domain": [None, 0, 1, 2, 3], "applicability_domain": ["Y", "N", "U"],
            "model_call_required": True, "semantic_label_fixed": False,
            "unchanged": "original packet/query, frozen controls/expectations, production no-document deterministic unknown path"}
