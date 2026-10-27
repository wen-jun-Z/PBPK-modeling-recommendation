"""Syntax-only JSON decoding for source-grounded PBPK assessment.

This grammar never chooses relevance, applicability or supporting excerpts. The
existing answer parser and source-scope validator must still run on every output.
Build the tokenizer index once per loaded judge; create a fresh batch callback for
each ``generate`` invocation, including repairs. No model or GPU is loaded here.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any, Sequence


DECODING_VERSION = "pbpk-source-judge-json-lmfe-v1"
MAX_CONSECUTIVE_WHITESPACES = 2


def judge_json_schema(request_count: int, reference_count: int) -> dict[str, Any]:
    """Permit every grade/applicability value and only supplied reference IDs.

    Unknown answers with an empty citation list remain possible. Known answers
    without citations and duplicate IDs are intentionally left for semantic
    validation, which can request a repair instead of inventing a citation.
    Empty-document packets use the runner's deterministic unknown path.
    """
    if type(request_count) is not int or request_count < 1:
        raise ValueError("request_count must be a positive integer")
    if type(reference_count) is not int or reference_count < 1:
        raise ValueError("No supplied excerpts: record unknown without generation")
    return {
        "type": "array",
        "minItems": request_count,
        "maxItems": request_count,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["i", "g", "a", "e"],
            "properties": {
                "i": {"type": "integer", "enum": list(range(1, request_count + 1))},
                "g": {"anyOf": [
                    {"type": "integer", "enum": [0, 1, 2, 3]},
                    {"type": "null"},
                ]},
                "a": {"type": "string", "enum": ["Y", "N", "U"]},
                "e": {
                    "type": "array", "minItems": 0, "maxItems": 3,
                    "items": {"type": "integer", "enum": list(range(1, reference_count + 1))},
                },
            },
        },
    }


def decoding_protocol() -> dict[str, Any]:
    """Nonsecret reproducibility fields to include in the frozen judge protocol."""
    # This template plus the supplied request/ref counts defines every grammar.
    template = judge_json_schema(1, 1)
    return {
        "version": DECODING_VERSION,
        "library": "lm-format-enforcer",
        "library_version": version("lm-format-enforcer"),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "schema_template": template,
        "schema_template_sha256": hashlib.sha256(
            json.dumps(template, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "request_id_enum": "1..request_count; uniqueness checked after decoding",
        "evidence_id_enum": "1..supplied_excerpt_count; no citation inferred",
        "max_consecutive_whitespaces": MAX_CONSECUTIVE_WHITESPACES,
        "force_json_field_order": False,
        "eos_policy": "all official model generation_config EOS IDs; only when JSON can end",
        "state_policy": "fresh parser per batch row and per generation; tokenizer index reused",
        "semantic_validation_required": True,
        "semantic_correctness_guaranteed": False,
        "token_limit_may_truncate": True,
    }


class JudgeJSONConstraint:
    """Reusable tokenizer data and isolated HF prefix callbacks for each batch."""

    def __init__(self, tokenizer: Any, eos_token_ids: int | Sequence[int]):
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

        if type(eos_token_ids) is int:
            eos_token_ids = [eos_token_ids]
        if not isinstance(eos_token_ids, (list, tuple)) or not eos_token_ids:
            raise ValueError("Official model EOS IDs are required")
        if any(type(value) is not int or value < 0 or value >= len(tokenizer) for value in eos_token_ids):
            raise ValueError("EOS IDs must be integer IDs in the tokenizer vocabulary")
        self.eos_token_ids = list(dict.fromkeys(eos_token_ids))
        self.tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
        # LMFE supports a list, while its HF builder defaults to tokenizer's
        # single EOS. Preserve the model's official additional stop token(s).
        self.tokenizer_data.eos_token_id = self.eos_token_ids

    def for_tasks(self, tasks: Sequence[tuple[dict[str, Any], Sequence[Any]]]):
        """Return the callback to pass as ``prefix_allowed_tokens_fn`` to HF.

        ``tasks`` is the existing runner's ``[(packet, queries), ...]``. HF's
        batch_id selects its own packet grammar even when padded prompts match.
        """
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

        if not tasks:
            raise ValueError("Cannot constrain an empty generation batch")
        callbacks = []
        for packet, queries in tasks:
            schema = judge_json_schema(len(queries), len(packet["refs"]))
            parser = JsonSchemaParser(schema, config=CharacterLevelParserConfig(
                max_consecutive_whitespaces=MAX_CONSECUTIVE_WHITESPACES,
                force_json_field_order=False,
                max_json_array_length=max(3, len(queries)),
            ))
            callbacks.append(build_transformers_prefix_allowed_tokens_fn(self.tokenizer_data, parser))

        def allowed_tokens(batch_id, token_ids):
            if type(batch_id) is not int or not 0 <= batch_id < len(callbacks):
                raise ValueError("Unknown generation batch row")
            return callbacks[batch_id](batch_id, token_ids)

        return allowed_tokens
