from __future__ import annotations

from pbpk_kb.recommendation.schemas import (
    DecisionReasonCode,
    RankedModelCandidate,
    RequirementProfile,
)


ABSTENTION_POLICY_VERSION = "pbpk-model-abstention-policy-1.0.0"


def abstention_reasons(
    *,
    profile: RequirementProfile,
    candidates: list[RankedModelCandidate],
    auto_reuse_enabled: bool,
) -> list[DecisionReasonCode]:
    reasons: set[DecisionReasonCode] = set()
    if profile.critical_gaps:
        reasons.add("CRITICAL_EVIDENCE_GAP")
    if any(conflict.severity == "hard" for conflict in profile.conflicts):
        reasons.add("UNRESOLVED_CONFLICT")
    reusable_candidates = [
        candidate
        for candidate in candidates
        if candidate.candidate_class
        in {
            "exact_compound_model",
            "same_compound_adaptation",
        }
    ]
    eligible = [
        candidate
        for candidate in reusable_candidates
        if candidate.eligibility == "eligible"
    ]
    if not eligible:
        reasons.add("NO_ELIGIBLE_MODEL")
        for candidate in reusable_candidates:
            reasons.update(candidate.exclusion_reason_codes)
    if not auto_reuse_enabled:
        reasons.add("TRAINING_LABELS_INSUFFICIENT")
    if eligible and not auto_reuse_enabled:
        reasons.add("SCORE_BELOW_THRESHOLD")
    return sorted(reasons)
