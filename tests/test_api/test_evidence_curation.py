import pytest
from fastapi import HTTPException

from pbpk_kb.api.evidence_routes import (
    CurationAction,
    modeling_approval_quality_issues,
    normalize_field_updates,
    review_queue,
    resolve_evidence_table,
    target_status_for_action,
    validate_modeling_approval_quality,
    validate_transition,
)


class _EmptyMappingsResult:
    def mappings(self):
        return []


class _CapturingSession:
    def __init__(self) -> None:
        self.sql = ""

    def execute(self, statement, params):
        self.sql = str(statement)
        return _EmptyMappingsResult()


def test_resolve_evidence_table_accepts_type_and_table_name() -> None:
    by_type = resolve_evidence_table("clinical_pk")
    by_table = resolve_evidence_table("clinical_pk_evidence")
    tissue = resolve_evidence_table("tissue_expression")
    model_parameter = resolve_evidence_table("model_parameter")

    assert by_type.table_name == "clinical_pk_evidence"
    assert by_table.evidence_type == "clinical_pk"
    assert tissue.table_name == "tissue_expression_evidence"
    assert model_parameter.table_name == "model_parameter_evidence"


def test_target_status_for_review_actions() -> None:
    assert target_status_for_action(CurationAction.accept, "needs_review") == "reviewed_accepted"
    assert target_status_for_action(CurationAction.reject, "needs_review") == "reviewed_rejected"
    assert target_status_for_action(CurationAction.approve_for_modeling, "reviewed_accepted") == "approved_for_modeling"
    assert target_status_for_action(CurationAction.edit, "needs_review") == "needs_review"


def test_validate_transition_blocks_skipping_review() -> None:
    with pytest.raises(HTTPException) as exc:
        validate_transition("machine_extracted", "reviewed_accepted", CurationAction.accept)

    assert exc.value.status_code == 409


def test_validate_transition_allows_accept_from_needs_review() -> None:
    validate_transition("needs_review", "reviewed_accepted", CurationAction.accept)


def test_normalize_field_updates_rejects_non_editable_fields() -> None:
    config = resolve_evidence_table("clinical_pk")

    with pytest.raises(HTTPException) as exc:
        normalize_field_updates(config, {"source_id": "pubchem"})

    assert exc.value.status_code == 400
    assert "source_id" in exc.value.detail["invalid_fields"]


def test_normalize_field_updates_allows_clinical_pk_value_fields() -> None:
    config = resolve_evidence_table("clinical_pk")

    updates = normalize_field_updates(config, {"value_text": "9.3 to 11.0", "unit": "mL/min/kg"})

    assert updates == {"value_text": "9.3 to 11.0", "unit": "mL/min/kg"}


def test_modeling_approval_quality_blocks_numeric_value_without_unit() -> None:
    config = resolve_evidence_table("clinical_pk")

    with pytest.raises(HTTPException) as exc:
        validate_modeling_approval_quality(
            config,
            {
                "value": 1.5,
                "value_text": None,
                "unit": None,
                "study_context": "single dose PK",
                "evidence_text": "The half-life was 1.5.",
                "source_location": "SECTION:PK",
            },
        )

    assert exc.value.status_code == 422
    assert any("numeric evidence requires a unit" in issue for issue in exc.value.detail["quality_issues"])


def test_modeling_approval_quality_blocks_value_less_candidates() -> None:
    config = resolve_evidence_table("clinical_pk")

    issues = modeling_approval_quality_issues(
        config,
        {
            "value": None,
            "value_text": None,
            "unit": None,
            "study_context": "Clinical Pharmacology section candidate",
            "evidence_text": "This section may contain PK text.",
            "source_location": "SECTION:Clinical Pharmacology",
        },
    )

    assert "approved_for_modeling requires a numeric value, value text, ratio, or equivalent model-ready value" in issues


def test_modeling_approval_quality_allows_curated_value_with_unit_and_context() -> None:
    config = resolve_evidence_table("clinical_pk")

    validate_modeling_approval_quality(
        config,
        {
            "value": None,
            "value_text": "9.3 to 11.0",
            "unit": "mL/min/kg",
            "study_context": "0.15 mg/kg IV dose",
            "evidence_text": "The mean total clearance ranged from 9.3 to 11.0 mL/min/kg.",
            "source_location": "SECTION:Elimination",
        },
    )


def test_review_queue_reads_reviewable_layer() -> None:
    db = _CapturingSession()

    rows = review_queue(
        status="needs_review",
        evidence_type="",
        source_id="",
        compound="",
        q="",
        min_confidence=None,
        created_from=None,
        created_to=None,
        duplicate_risk="all",
        missing_unit=None,
        missing_context=None,
        limit=50,
        db=db,
    )

    assert rows == []
    assert "FROM reviewable_evidence e" in db.sql
    assert "FROM auto_validated_evidence e" not in db.sql
