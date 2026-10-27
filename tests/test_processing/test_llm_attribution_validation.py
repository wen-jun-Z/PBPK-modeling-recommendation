from __future__ import annotations

# [FIX 2026-07-02 · claude-opus-4-8] (F1) numeric identifiers must not substring-validate a compound attribution.

from pbpk_kb.processing.llm_attribution_file import (
    compound_name_terms,
    validate_candidate,
)

_COMPOUND = {
    "compound_id": "c1",
    "preferred_name": "metformin",
    "pubchem_cid": "4091",
    "rxnorm_rxcui": "6809",
    "synonyms": ["Glucophage"],
}


def test_compound_name_terms_excludes_numeric_ids() -> None:
    terms = compound_name_terms(_COMPOUND)
    assert "metformin" in terms
    assert "Glucophage" in terms
    assert "4091" not in terms
    assert "6809" not in terms


def test_stray_number_does_not_validate_attribution() -> None:
    # matched_text is copied from source_text but contains only the RxCUI digits, not the compound name.
    row = {"evidence_id": "e1", "source_text": "The study id was 6809 in the analysis cohort."}
    candidate = {"matched_text": "6809", "compound_role": "target_compound", "compound_name": "metformin"}
    accepted, reason, refined = validate_candidate(row, [_COMPOUND], candidate)
    assert accepted is False
    assert reason == "matched_text_does_not_contain_selected_compound"
    assert refined is None


def test_real_name_mention_still_validates() -> None:
    row = {"evidence_id": "e2", "source_text": "Metformin 500 mg was administered to healthy adults."}
    candidate = {"matched_text": "Metformin 500 mg", "compound_role": "target_compound", "compound_name": "metformin"}
    accepted, reason, refined = validate_candidate(row, [_COMPOUND], candidate)
    assert accepted is True
    assert refined and refined["refined_compound_name"] == "metformin"
