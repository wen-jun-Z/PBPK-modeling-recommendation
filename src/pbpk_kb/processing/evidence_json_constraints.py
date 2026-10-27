"""Process-local JSON syntax corrections for the project evidence decoder.

The upstream dependency and the scientific output schemas are left unchanged.
"""
from __future__ import annotations

import hashlib
from contextvars import ContextVar
from pathlib import Path


OBJECT_SEPARATOR_COMPATIBILITY = "strict_object_separators_v1"
UNICODE_FREETEXT_COMPATIBILITY = "unicode_freetext_consumption_v2"
_LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_CONSUMED_UNICODE: ContextVar[str | None] = ContextVar("evidence_json_consumed_unicode", default=None)


def object_separator_compatibility() -> dict[str, str]:
    """Describe the code loaded by this process, not later on-disk edits."""
    return {"schema_object_separator_compatibility": OBJECT_SEPARATOR_COMPATIBILITY,
            "schema_object_separator_patch_sha256": _LOADED_SOURCE_SHA256}


def unicode_freetext_compatibility() -> dict[str, str]:
    return {"schema_unicode_compatibility": UNICODE_FREETEXT_COMPATIBILITY,
            "schema_unicode_patch_sha256": _LOADED_SOURCE_SHA256}


def install_unicode_freetext_consumption() -> None:
    """Keep byte-fallback Unicode inside its unfinished JSON string.

    LMFE's free-text token cache accepts legal Unicode not present in the
    tokenizer's single-token alphabet. On consumption, its JSON parser instead
    discards every stack frame that cannot find that character in the alphabet,
    incorrectly permitting EOS. Only during consumption, expose this one legal
    character to an open, unconstrained-by-enum/regex string with length room.
    Token enumeration and the native free-text cache stay unchanged. ContextVar
    scopes nested calls and concurrent callbacks without mutating any alphabet,
    schema or parser branch. Quotes, escapes, controls and surrogates do not use
    this compatibility path.
    """
    from lmformatenforcer import jsonschemaparser as module

    original_add = module.JsonSchemaParser.add_character
    if getattr(original_add, "pbpk_unicode_freetext_consumption", False):
        return
    original_allowed = module.StringParsingState.get_allowed_characters

    def allowed_characters(self):
        allowed = original_allowed(self)
        character = _CONSUMED_UNICODE.get()
        if (
            character is not None and character not in allowed
            and self.seen_opening_quote and not self.seen_closing_quote
            and not self.allowed_strings and not self.regex_parser
            and (self.max_length is None or len(self.parsed_string) < self.max_length)
        ):
            return allowed + character
        return allowed

    def add_character(self, character):
        if (len(character) != 1 or ord(character) < 128
                or 0xD800 <= ord(character) <= 0xDFFF):
            return original_add(self, character)
        token = _CONSUMED_UNICODE.set(character)
        try:
            return original_add(self, character)
        finally:
            _CONSUMED_UNICODE.reset(token)

    add_character.pbpk_unicode_freetext_consumption = True
    module.StringParsingState.get_allowed_characters = allowed_characters
    module.JsonSchemaParser.add_character = add_character


def install_strict_object_separators() -> None:
    """Disallow a closing brace immediately after an object separator.

    LMFE 0.11.3 uses the same state for an empty object's first key and the key
    following a comma. It therefore permits ``{"optional": 1,}`` once all
    required properties are present. An object in that state with existing
    keys must read another key before closing; an initially empty object may
    still close when its schema permits it. Other states, string parsing and
    the bounded free-text token cache are untouched.
    """
    from lmformatenforcer import jsonschemaparser as module

    original = module.ObjectParsingState.get_allowed_characters
    if getattr(original, "pbpk_strict_object_separators", False):
        return

    def strict_allowed_characters(self):
        allowed = original(self)
        if (self.current_stage == module.ObjectParsingStage.PARSING_KEY_OR_END
                and self.existing_keys):
            return allowed.replace("}", "")
        return allowed

    strict_allowed_characters.pbpk_strict_object_separators = True
    module.ObjectParsingState.get_allowed_characters = strict_allowed_characters
