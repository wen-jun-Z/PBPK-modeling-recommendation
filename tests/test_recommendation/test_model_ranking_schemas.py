from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from pbpk_kb.recommendation.schemas import (
    ConstraintCheck,
    RankedModelCandidate,
    RankedModelRecommendation,
    ranked_model_recommendation_checksum,
)


MODEL_ID = UUID("52163da6-a71c-4aaa-b6ff-0592c3f559e4")
PROFILE_ID = UUID("9f81dd42-ab7f-4b6f-a090-ff01cc742efe")
REQUIREMENT_ID = UUID("ec846a2e-ee7a-49b7-b208-2e54bf12d91a")
RUN_ID = UUID("16b5ce97-2640-4383-9358-b40fd3cbe646")
MANIFEST_ID = UUID("35911bbe-bc71-48e6-bb75-6a3b2e4f9dd8")


def _blocked_candidate() -> RankedModelCandidate:
    return RankedModelCandidate(
        model_version_id=MODEL_ID,
        source_model_version_key="model-version:source-key",
        model_family="Midazolam",
        candidate_class="exact_compound_model",
        eligibility="ineligible",
        constraint_checks=[
            ConstraintCheck(
                constraint_id="license",
                model_version_id=MODEL_ID,
                source_model_version_key="model-version:source-key",
                requirement_id=REQUIREMENT_ID,
                severity="hard",
                result="fail",
                observed_value="license_unresolved",
                reason_code="LICENSE_UNRESOLVED",
                model_asset_locations=[
                    "GITHUB_BLOB:commit/Midazolam/model.pksim5"
                ],
            )
        ],
        uncertainty=1.0,
        unmet_requirement_ids=[REQUIREMENT_ID],
        exclusion_reason_codes=["LICENSE_UNRESOLVED"],
    )


def test_abstain_contract_is_checksum_valid() -> None:
    payload = {
        "ranking_run_id": RUN_ID,
        "ranking_schema_version": "1.1.0",
        "profile_id": PROFILE_ID,
        "decision": "abstain",
        "candidates": [_blocked_candidate()],
        "selected_model_version_id": None,
        "adaptation_plan_id": None,
        "abstention_reason_codes": [
            "LICENSE_UNRESOLVED",
            "NO_ELIGIBLE_MODEL",
        ],
        "decision_certainty": 1.0,
        "confidence_semantics": "deterministic_gate",
        "model_run_manifest_id": MANIFEST_ID,
        "auto_reuse_enabled": False,
        "created_at": datetime.now(UTC),
    }
    draft = RankedModelRecommendation.model_construct(
        **payload,
        checksum="0" * 64,
    )
    payload["checksum"] = ranked_model_recommendation_checksum(
        draft.model_dump(mode="json")
    )

    recommendation = RankedModelRecommendation.model_validate(payload)

    assert recommendation.decision == "abstain"
    assert recommendation.selected_model_version_id is None


def test_reuse_rejects_hard_failed_candidate() -> None:
    payload = {
        "ranking_run_id": RUN_ID,
        "ranking_schema_version": "1.1.0",
        "profile_id": PROFILE_ID,
        "decision": "reuse",
        "candidates": [_blocked_candidate()],
        "selected_model_version_id": MODEL_ID,
        "adaptation_plan_id": None,
        "abstention_reason_codes": [],
        "decision_certainty": 1.0,
        "confidence_semantics": "deterministic_gate",
        "model_run_manifest_id": MANIFEST_ID,
        "auto_reuse_enabled": True,
        "created_at": datetime.now(UTC),
    }
    draft = RankedModelRecommendation.model_construct(
        **payload,
        checksum="0" * 64,
    )
    payload["checksum"] = ranked_model_recommendation_checksum(
        draft.model_dump(mode="json")
    )

    with pytest.raises(
        ValidationError,
        match="reuse requires one eligible selected model",
    ):
        RankedModelRecommendation.model_validate(payload)
