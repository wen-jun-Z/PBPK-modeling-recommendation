from __future__ import annotations

# Regression tests for extraction precision, including numeric-context,
# topic-only text, clinical endpoints, and value-to-parameter alignment.

from pbpk_kb.processing.raw_processor import (
    ADME_ACRONYM_TERMS,
    PBPK_TERMS,
    PK_PATTERNS,
    _looks_like_numeric_context,
    _normalize_parameter,
)


def _match_bioavailability(text: str):
    for parameter, pattern in PK_PATTERNS:
        if parameter != "bioavailability":
            continue
        m = pattern.search(text)
        if m:
            return m
    return None


def test_numeric_array_is_rejected_context() -> None:
    # The exact shape that produced metformin bioavailability=0 from a ChEMBL fingerprint array.
    assert _looks_like_numeric_context("0000 F 0 0 0 0 0 0 0 0 0 0 0 0\n 24.") is True


def test_prose_sentence_is_not_numeric_context() -> None:
    assert _looks_like_numeric_context("The absolute bioavailability was 41% after oral administration.") is False


def test_bare_F_no_longer_extracts_bioavailability_from_array() -> None:
    # D1: "F" surrounded by an array must NOT yield a bioavailability value.
    assert _match_bioavailability("0000 F 0 0 0 0 0 0 0 0") is None


def test_strict_F_shorthand_still_matches() -> None:
    # No "bioavailability" word here, so only the strict F=NN% pattern can match.
    m = _match_bioavailability("Systemic exposure indicated F = 41% in adults.")
    assert m is not None
    assert m.group(1) == "41"


def test_bioavailability_word_still_matches() -> None:
    m = _match_bioavailability("absolute bioavailability 53 %")
    assert m is not None
    assert m.group(1) == "53"


def test_adme_acronyms_are_case_sensitive_month_oct_excluded() -> None:
    # D4: month "Oct" must not match; real OCT1/OCT2 must match.
    assert ADME_ACRONYM_TERMS.search("submitted in Oct 2015") is None
    assert ADME_ACRONYM_TERMS.search("substrate of OCT2 transporter") is not None


def test_pbpk_terms_drop_generic_words() -> None:
    # D5: generic "model"/"simulation" no longer flag PBPK; real tokens still do.
    assert PBPK_TERMS.search("a linear regression model was fitted") is None
    assert PBPK_TERMS.search("a whole-body PBPK model in PK-Sim") is not None


def test_parameter_case_normalization() -> None:
    # D8: case-split duplicates collapse.
    assert _normalize_parameter("Metabolism") == "metabolism"
    assert _normalize_parameter("PREGNANCY") == "pregnancy"
    assert _normalize_parameter("cyp3a4") == "CYP3A4"
    assert _normalize_parameter("Oct2") == "OCT2"
