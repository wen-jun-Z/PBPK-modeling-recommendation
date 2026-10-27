from __future__ import annotations

import pytest

from pbpk_kb.processing.run_llm_extraction import (
    PendingTask,
    UpstreamIndex,
    _numeric_parameter_owner_error,
    _validate_candidates,
)


AMOXICILLIN_EXCRETION = (
    "Approximately 60% of the orally administered dose of amoxicillin is excreted "
    "unchanged in the urine within six to eight hours post-dose; its excretion can "
    "be delayed by concurrent administration of probenecid."
)


def _validate(source_text: str, **changes):
    chunk = {
        "run_id": "test-run", "source_id": "dailymed", "raw_asset_id": "asset-1",
        "storage_path": "/raw/test-label.xml", "checksum": "test-checksum",
        "source_location": "test-label#clinical-pharmacology", "chunk_id": "chunk-1",
        "parsed_document_id": "doc-1", "chunk_text": source_text,
    }
    task = PendingTask(
        chunk, "amoxicillin", {},
        {"request_id": "test-request", "input_items": [source_text]}, "test-input",
    )
    upstream = UpstreamIndex(
        raw_assets={"asset-1": chunk}, parsed_document_ids={"doc-1"},
        parsed_table_ids=set(), target_compounds={"amoxicillin"},
    )
    candidate = {
        "target_compound": "amoxicillin", "evidence_type": "adme",
        "parameter": "urinary_excretion_unchanged", "value": 60, "unit": "%",
        "condition": None, "source_text": source_text, "confidence": 0.9,
        "requires_review": True, **changes,
    }
    return _validate_candidates(task, [candidate], upstream)


@pytest.mark.parametrize("dose_phrase", [
    "orally administered dose", "intravenously administered dose", "administered dose", "dose",
])
def test_administered_dose_percentage_belongs_to_named_drug(dose_phrase):
    source = AMOXICILLIN_EXCRETION.replace("orally administered dose", dose_phrase)
    valid, rejected = _validate(source)
    assert not rejected
    assert valid[0]["value"] == 60
    assert valid[0]["unit"] == "%"


@pytest.mark.parametrize("owner", [
    "Comparatorin", "metabolites of amoxicillin", "the active metabolite of amoxicillin",
])
@pytest.mark.parametrize("dose_phrase", ["", "the orally administered dose of "])
def test_dose_percentage_of_other_drug_or_metabolite_stays_rejected(owner, dose_phrase):
    source = (
        f"Approximately 60% of {dose_phrase}{owner} is excreted unchanged in urine "
        "during coadministration with amoxicillin."
    )
    assert _numeric_parameter_owner_error(
        {"parameter": "urinary_excretion_unchanged", "source_text": source, "value": 60, "unit": "%"},
        "amoxicillin",
    ) == "numeric_value_belongs_to_other_substance"
    valid, rejected = _validate(source)
    assert not valid
    assert "numeric_value_belongs_to_other_substance" in rejected[0]["errors"]


def test_excretion_delay_is_supported_as_an_explicit_ddi_mechanism():
    valid, rejected = _validate(
        AMOXICILLIN_EXCRETION, evidence_type="ddi", parameter="excretion_delay",
        value=None, unit=None,
    )
    assert not rejected
    assert valid[0]["parameter"] == "excretion_delay"
    assert valid[0]["value"] is None


def test_excretion_delay_cannot_borrow_unchanged_urinary_fraction():
    valid, rejected = _validate(
        AMOXICILLIN_EXCRETION, evidence_type="ddi", parameter="excretion_delay",
    )
    assert not valid
    assert "numeric_value_not_supported_for_target_parameter" in rejected[0]["errors"]


def test_excretion_delay_requires_source_delay_and_does_not_open_generic_ddi_excretion():
    source = "Amoxicillin is excreted unchanged in urine during coadministration with probenecid."
    valid, rejected = _validate(
        source, evidence_type="ddi", parameter="excretion_delay", value=None, unit=None,
    )
    assert not valid
    assert "parameter_not_supported_by_source_text_terminology" in rejected[0]["errors"]
    valid, rejected = _validate(
        AMOXICILLIN_EXCRETION, evidence_type="ddi", parameter="urinary_excretion_unchanged",
    )
    assert not valid
    assert "parameter_outside_evidence_type_domain" in rejected[0]["errors"]


@pytest.mark.parametrize("parameter,accepted", [("AUC", True), ("Cmax", False)])
def test_fda_dot_auc_unit_keeps_parameter_dimension(parameter, accepted):
    valid, rejected = _validate(
        f"Amoxicillin {parameter} was 12 µg.hr/mL in healthy adults.",
        evidence_type="clinical_pk", parameter=parameter, value=12, unit="µg.hr/mL",
    )
    assert bool(valid) is accepted
    if not accepted:
        assert "unit_dimension_does_not_match_parameter" in rejected[0]["errors"]

