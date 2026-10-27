from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from pbpk_kb.model_assets.models import (
    ExecutionProbeStatus,
    LicenseStatus,
    OSPModelVersionSnapshot,
    ParseStatus,
)
from pbpk_kb.recommendation.model_constraints import (
    evaluate_model_constraints,
    model_version_uuid,
)
from pbpk_kb.recommendation.schemas import (
    ConstraintCheck,
    DecisionReasonCode,
    ModelScoreFeature,
    RankedModelCandidate,
    RequirementProfile,
)


CANDIDATE_RETRIEVAL_VERSION = "pbpk-model-candidate-retrieval-1.0.0"
FEATURE_METHOD_VERSION = "diagnostic-unweighted-coverage-1.0.0"
CANDIDATE_LAYER_ORDER = (
    "validated_reuse",
    "controlled_adaptation_candidate",
    "same_compound_qualification_candidate",
    "mechanistic_reference",
    "blocked",
)
CANDIDATE_LAYER_LABELS = {
    "validated_reuse": "Validated for reuse",
    "controlled_adaptation_candidate": (
        "Controlled adaptation candidate"
    ),
    "same_compound_qualification_candidate": (
        "Same-compound qualification candidate"
    ),
    "mechanistic_reference": "Mechanistic reference",
    "blocked": "Blocked",
}
CANDIDATE_LAYER_PERMITTED_USE = {
    "validated_reuse": (
        "Direct reuse within the registered model scope."
    ),
    "controlled_adaptation_candidate": (
        "Adapt only through a versioned plan, rerun, and independent "
        "validation; direct reuse is prohibited."
    ),
    "same_compound_qualification_candidate": (
        "Qualification work only; resolve license and validation gaps "
        "before adaptation or reuse."
    ),
    "mechanistic_reference": (
        "Structure and mechanism reference only; compound-specific "
        "parameters must not be transferred."
    ),
    "blocked": (
        "Metadata and failure-audit retention only; exclude from reuse "
        "and adaptation."
    ),
}
_TECHNICAL_BLOCKER_CODES = frozenset(
    {
        "ASSET_MISSING",
        "CHECKSUM_FAILED",
        "PARSER_FAILED",
        "EXECUTION_PROBE_FAILED",
        "SOFTWARE_UNAVAILABLE",
    }
)
_CONTEXT_CONSTRAINT_MARKERS = (
    ":species:",
    ":route:",
    ":formulation:",
    ":dose_regimen_scope:",
    ":explicit_request_constraint:",
    ":enzyme:",
    ":transporter:",
    ":compartment:",
    ":adme_process:",
    ":prediction_endpoint:",
    ":population:",
    ":ddi_role:",
    ":model_structure:",
)


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _check_value(
    checks: Iterable[ConstraintCheck],
    markers: tuple[str, ...],
) -> float:
    selected = [
        check
        for check in checks
        if any(marker in check.constraint_id for marker in markers)
    ]
    if not selected:
        return 1.0
    return sum(check.result == "pass" for check in selected) / len(
        selected
    )


def _feature(
    name: str,
    value: float,
) -> ModelScoreFeature:
    normalized = min(1.0, max(0.0, float(value)))
    return ModelScoreFeature(
        feature_name=name,
        raw_value=float(value),
        normalized_value=normalized,
        weight=0.0,
        weighted_value=0.0,
        method_version=FEATURE_METHOD_VERSION,
    )


def _diagnostic_features(
    *,
    profile: RequirementProfile,
    model: OSPModelVersionSnapshot,
    checks: list[ConstraintCheck],
) -> list[ModelScoreFeature]:
    context_failures = sum(
        check.result in {"fail", "unknown"}
        and any(
            marker in check.constraint_id
            for marker in _CONTEXT_CONSTRAINT_MARKERS
        )
        for check in checks
    )
    context_total = sum(
        any(
            marker in check.constraint_id
            for marker in _CONTEXT_CONSTRAINT_MARKERS
        )
        for check in checks
    )
    hard_checks = [
        check for check in checks if check.severity == "hard"
    ]
    hard_unknown = sum(
        check.result == "unknown" for check in hard_checks
    )
    unresolved_license = any(
        check.reason_code == "LICENSE_UNRESOLVED"
        and check.result != "pass"
        for check in checks
    )
    uncertainty = (
        1.0
        if unresolved_license
        else hard_unknown / len(hard_checks)
        if hard_checks
        else 0.0
    )
    population_requested = any(
        requirement.category == "population_feature"
        and requirement.origin == "user_request"
        for requirement in profile.requirements
    )
    specialized_population = any(
        marker in " ".join(
            [model.project_relative_path, *model.populations]
        ).casefold()
        for marker in (
            "pediatric",
            "paediatric",
            "neonat",
            "infant",
            "pregnan",
            "renal impairment",
            "hepatic impairment",
        )
    )
    population_compatibility = (
        _check_value(checks, (":population:",))
        if population_requested
        else 0.6 if specialized_population else 1.0
    )
    return [
        _feature(
            "compound_identity_match",
            _check_value(checks, (":compound_identity:",)),
        ),
        _feature(
            "mechanism_coverage",
            _check_value(
                checks,
                (
                    ":enzyme:",
                    ":transporter:",
                    ":adme_process:",
                    ":ddi_role:",
                ),
            ),
        ),
        _feature(
            "route_formulation_match",
            _check_value(
                checks,
                (":route:", ":formulation:"),
            ),
        ),
        _feature(
            "population_match",
            population_compatibility,
        ),
        _feature(
            "structure_match",
            _check_value(
                checks,
                (":model_structure:", ":compartment:"),
            ),
        ),
        _feature(
            "endpoint_coverage",
            _check_value(checks, (":prediction_endpoint:",)),
        ),
        _feature(
            "validation_quality",
            min(1.0, model.validation_metric_count / 3.0),
        ),
        _feature(
            "validation_transferability",
            float(
                model.validation_metric_count > 0
                and model.dataset_reference_count > 0
            ),
        ),
        _feature("source_quality", 1.0),
        _feature(
            "reproducibility",
            _check_value(checks, (":execution_probe:",)),
        ),
        _feature(
            "parameter_completeness",
            min(
                1.0,
                math.log10(model.parameter_record_count + 1) / 6.0,
            ),
        ),
        _feature(
            "evidence_conflict_penalty",
            float(bool(profile.conflicts)),
        ),
        _feature(
            "adaptation_cost",
            context_failures / context_total
            if context_total
            else 0.0,
        ),
        _feature("uncertainty", uncertainty),
    ]


def _candidate_class(
    profile: RequirementProfile,
    model: OSPModelVersionSnapshot,
    checks: list[ConstraintCheck],
) -> str:
    identity = next(
        check
        for check in checks
        if ":compound_identity:" in check.constraint_id
    )
    if identity.result != "pass":
        return "mechanistic_template"
    requested_names = {
        " ".join(value.casefold().split())
        for value in (
            profile.compound_identity.preferred_name,
            profile.compound_identity.active_moiety,
        )
    }
    model_family = " ".join(model.model_family.casefold().split())
    if model_family not in requested_names:
        # A compound mentioned inside a project can be a DDI probe, comparator,
        # metabolite, or validation analyte. Only the model family's primary
        # active moiety can enter a same-compound reuse/adaptation layer.
        return "mechanistic_template"
    has_context_failure = any(
        check.result in {"fail", "unknown"}
        and any(
            marker in check.constraint_id
            for marker in _CONTEXT_CONSTRAINT_MARKERS
            if marker != ":compound_identity:"
        )
        for check in checks
    )
    return (
        "same_compound_adaptation"
        if has_context_failure
        else "exact_compound_model"
    )


def build_ranked_candidate(
    *,
    profile: RequirementProfile,
    model: OSPModelVersionSnapshot,
) -> RankedModelCandidate:
    checks = evaluate_model_constraints(
        profile=profile,
        model=model,
    )
    candidate_class = _candidate_class(profile, model, checks)
    hard_blocked = any(
        check.severity == "hard"
        and check.result in {"fail", "unknown"}
        for check in checks
    )
    eligibility = (
        "template_only"
        if candidate_class == "mechanistic_template"
        else "ineligible"
        if hard_blocked
        else "eligible"
    )
    grouped: dict[object, list[ConstraintCheck]] = defaultdict(list)
    for check in checks:
        grouped[check.requirement_id].append(check)
    unmet = {
        requirement_id
        for requirement_id, values in grouped.items()
        if any(
            value.severity == "hard"
            and value.result in {"fail", "unknown"}
            for value in values
        )
    }
    matched = {
        requirement_id
        for requirement_id, values in grouped.items()
        if requirement_id not in unmet
        and any(value.result == "pass" for value in values)
    }
    reason_codes = sorted(
        {
            check.reason_code
            for check in checks
            if check.result in {"fail", "unknown"}
        }
    )
    features = _diagnostic_features(
        profile=profile,
        model=model,
        checks=checks,
    )
    positive_features = [
        feature.normalized_value
        for feature in features
        if feature.feature_name
        not in {
            "evidence_conflict_penalty",
            "adaptation_cost",
            "uncertainty",
        }
    ]
    deterministic_compatibility = (
        sum(positive_features) / len(positive_features)
        if positive_features
        else 0.0
    )
    uncertainty = next(
        feature.normalized_value
        for feature in features
        if feature.feature_name == "uncertainty"
    )
    return RankedModelCandidate(
        model_version_id=model_version_uuid(model.model_version_id),
        source_model_version_key=model.model_version_id,
        model_family=model.model_family,
        candidate_class=candidate_class,
        eligibility=eligibility,
        constraint_checks=checks,
        score_features=features,
        deterministic_compatibility=deterministic_compatibility,
        learned_graph_compatibility=None,
        final_score=None,
        uncertainty=uncertainty,
        validation_run_id=(
            model.validation_run_label.validation_run_id
            if model.validation_run_label is not None
            else None
        ),
        run_label_grade=(
            model.validation_run_label.run_label_grade
            if model.validation_run_label is not None
            else 0
        ),
        ranking_label=(
            model.validation_run_label.ranking_label
            if model.validation_run_label is not None
            else None
        ),
        matched_requirement_ids=sorted(matched, key=str),
        unmet_requirement_ids=sorted(unmet, key=str),
        exclusion_reason_codes=reason_codes,
    )


def _qualification_actions(
    *,
    profile: RequirementProfile,
    model: OSPModelVersionSnapshot,
    candidate: RankedModelCandidate,
    layer: str,
) -> list[str]:
    if layer == "validated_reuse":
        return ["reuse_within_registered_scope"]
    if layer == "mechanistic_reference":
        return [
            "use_for_structure_and_mechanism_reference_only",
            "do_not_transfer_compound_specific_parameters",
        ]

    failed_reason_codes = {
        check.reason_code
        for check in candidate.constraint_checks
        if check.severity == "hard"
        and check.result in {"fail", "unknown"}
    }
    actions: list[str] = []
    if layer == "blocked":
        actions.extend(
            [
                "retain_failure_audit",
                "exclude_from_reuse_and_adaptation",
            ]
        )
    if model.license.status == LicenseStatus.UNRESOLVED:
        actions.append("resolve_model_license")
    elif model.license.status == LicenseStatus.RESTRICTED:
        actions.append("record_license_restriction")
    if model.static_parse_status == ParseStatus.REGISTERED_NOT_EXECUTABLE and (
        model.pkml_parse_status == ParseStatus.PARSED
        and model.execution_probe_status == ExecutionProbeStatus.PASSED
    ):
        actions.append("reconcile_static_parse_status")
    if "ASSET_MISSING" in failed_reason_codes:
        actions.append("restore_and_checksum_model_assets")
    if "CHECKSUM_FAILED" in failed_reason_codes:
        actions.append("resolve_asset_checksum_mismatch")
    if "PARSER_FAILED" in failed_reason_codes:
        actions.append("repair_or_add_format_parser")
    if "EXECUTION_PROBE_FAILED" in failed_reason_codes:
        actions.append("rerun_runtime_execution_probe")
    if "SOFTWARE_UNAVAILABLE" in failed_reason_codes:
        actions.append("provision_locked_runtime")
    if candidate.validation_run_id is None:
        actions.extend(
            [
                "register_preregistered_validation_protocol",
                "complete_independent_external_validation",
            ]
        )
    elif candidate.run_label_grade < 3:
        actions.append("resolve_validation_failure_under_new_protocol")
    if "CRITICAL_EVIDENCE_GAP" in failed_reason_codes:
        actions.append("resolve_critical_requirement_evidence_gaps")
    if "UNRESOLVED_CONFLICT" in failed_reason_codes:
        actions.append("resolve_requirement_conflicts")
    if layer == "controlled_adaptation_candidate":
        actions.extend(
            [
                "create_versioned_adaptation_plan",
                "rerun_all_required_simulations",
                "independently_validate_adapted_version",
            ]
        )

    # Keep actions stable so API checksums and UI ordering are reproducible.
    return list(dict.fromkeys(actions))


def candidate_layer_decision(
    *,
    profile: RequirementProfile,
    model: OSPModelVersionSnapshot,
    candidate: RankedModelCandidate,
    relevance_decision: dict[str, object] | None = None,
) -> dict[str, object]:
    """Assign a non-promotional use layer to a retrieved model version."""

    if candidate.candidate_class == "mechanistic_template" and not (
        relevance_decision
        and relevance_decision.get("is_relevant") is True
        and relevance_decision.get("relevance_class")
        == "shared_verified_mechanism"
    ):
        raise ValueError(
            "A different-compound model cannot enter a candidate layer "
            "without a verified shared-mechanism relevance decision."
        )

    hard_failures = {
        check.reason_code
        for check in candidate.constraint_checks
        if check.severity == "hard"
        and check.result in {"fail", "unknown"}
    }
    technical_blockers = sorted(
        hard_failures & _TECHNICAL_BLOCKER_CODES
    )
    if model.license.status == LicenseStatus.RESTRICTED:
        technical_blockers = sorted(
            {*technical_blockers, "LICENSE_RESTRICTED"}
        )

    requested_names = {
        _normalize_name(profile.compound_identity.preferred_name),
        _normalize_name(profile.compound_identity.active_moiety),
    }
    exact_family = _normalize_name(model.model_family) in requested_names
    validated_reuse = bool(
        candidate.candidate_class == "exact_compound_model"
        and candidate.eligibility == "eligible"
        and candidate.run_label_grade == 3
        and candidate.ranking_label == "reuse_positive"
        and model.reusable_candidate
    )

    if technical_blockers:
        layer = "blocked"
    elif candidate.candidate_class == "mechanistic_template":
        layer = "mechanistic_reference"
    elif validated_reuse:
        layer = "validated_reuse"
    elif (
        exact_family
        and candidate.candidate_class == "same_compound_adaptation"
        and model.license.status == LicenseStatus.ALLOWED
        and model.pkml_parse_status == ParseStatus.PARSED
        and model.execution_probe_status == ExecutionProbeStatus.PASSED
    ):
        layer = "controlled_adaptation_candidate"
    else:
        layer = "same_compound_qualification_candidate"

    actions = _qualification_actions(
        profile=profile,
        model=model,
        candidate=candidate,
        layer=layer,
    )
    return {
        "candidate_layer": layer,
        "candidate_layer_label": CANDIDATE_LAYER_LABELS[layer],
        "candidate_layer_order": CANDIDATE_LAYER_ORDER.index(layer),
        "permitted_use": CANDIDATE_LAYER_PERMITTED_USE[layer],
        "direct_reuse_allowed": layer == "validated_reuse",
        "technical_blocker_codes": technical_blockers,
        "qualification_actions": actions,
    }


def retrieve_model_candidates(
    *,
    profile: RequirementProfile,
    models: Iterable[OSPModelVersionSnapshot],
) -> list[RankedModelCandidate]:
    candidates = [
        build_ranked_candidate(
            profile=profile,
            model=model,
        )
        for model in models
    ]
    class_order = {
        "exact_compound_model": 0,
        "same_compound_adaptation": 1,
        "mechanistic_template": 2,
    }
    candidates.sort(
        key=lambda candidate: (
            class_order[candidate.candidate_class],
            candidate.eligibility != "eligible",
            -float(candidate.deterministic_compatibility or 0.0),
            candidate.model_family.casefold(),
            str(candidate.model_version_id),
        )
    )
    return candidates
