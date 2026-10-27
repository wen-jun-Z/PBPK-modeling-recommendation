from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from pbpk_kb.recommendation.schemas import (
    CompoundRef,
    ModelingRequest,
    RecommendationResponse,
    RequirementConflictRef,
    RequirementContext,
    RequirementGap,
    RequirementItem,
    RequirementProfile,
    requirement_profile_checksum,
)


PROFILE_VERSION = "1.0.0"
PROFILE_BUILDER_VERSION = "pbpk-requirement-profile-builder-1.0.0"
PROFILE_NAMESPACE = UUID("878646e6-4155-48c7-972c-127caf89c307")
REQUIREMENT_NAMESPACE = UUID("b20dc4a6-ae0a-4b64-a998-8488ba7d8ba5")
ONTOLOGY_RULE_NAMESPACE = UUID("7d447b13-e3e8-46bf-b6a2-4a52fe708294")
REQUEST_NAMESPACE = UUID("59d2eb31-a85e-41b5-b842-07f0bb9f8405")

NECESSITY_PRIORITY = {
    "unsupported": 0,
    "optional": 1,
    "preferred": 2,
    "hard_required": 3,
}
OBJECTIVE_PROCESS_RULES: dict[str, tuple[str, ...]] = {
    "base_pk": ("absorption", "distribution", "metabolism", "excretion"),
    "first_in_human": (
        "absorption",
        "distribution",
        "metabolism",
        "excretion",
    ),
    "dose_selection": (
        "absorption",
        "distribution",
        "metabolism",
        "excretion",
    ),
    "formulation_comparison": ("absorption", "distribution"),
    "food_effect": ("absorption",),
    "bioequivalence": ("absorption", "distribution", "excretion"),
    "ddi_victim": ("metabolism", "excretion", "ddi"),
    "ddi_perpetrator": ("metabolism", "ddi"),
    "renal_impairment": ("distribution", "metabolism", "renal_excretion"),
    "hepatic_impairment": ("distribution", "metabolism", "excretion"),
    "pediatric": ("absorption", "distribution", "metabolism", "excretion"),
    "geriatric": ("absorption", "distribution", "metabolism", "excretion"),
    "pregnancy": ("absorption", "distribution", "metabolism", "excretion"),
    "pharmacogenomics": ("metabolism",),
    "tissue_exposure": ("distribution",),
    "public_model_reuse": (
        "absorption",
        "distribution",
        "metabolism",
        "excretion",
    ),
    "oral_absorption": ("absorption", "distribution"),
    "ddi": ("metabolism", "excretion", "ddi"),
    "special_population": (
        "absorption",
        "distribution",
        "metabolism",
        "excretion",
    ),
}
COMPONENT_TO_REQUIREMENT = {
    "physicochemical": ("parameter", "physicochemical_parameter_coverage"),
    "absorption": ("adme_process", "absorption"),
    "distribution": ("adme_process", "distribution"),
    "metabolism": ("adme_process", "metabolism"),
    "transport": ("adme_process", "transport"),
    "excretion": ("adme_process", "excretion"),
    "ddi": ("adme_process", "ddi"),
    "special_population": (
        "population_feature",
        "special_population_support",
    ),
    "public_model_template": (
        "model_structure",
        "public_model_template",
    ),
}


class RequirementProfileBuildError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _stable_uuid(namespace: UUID, *parts: Any) -> UUID:
    return uuid5(namespace, "\x1f".join(_canonical_json(part) for part in parts))


def request_uuid(request: ModelingRequest) -> UUID:
    return _stable_uuid(
        REQUEST_NAMESPACE,
        request.model_dump(mode="json"),
    )


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _validated_uuid(value: str, *, field: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise RequirementProfileBuildError(
            f"{field} must reference a UUID, received {value!r}."
        ) from exc


def _ontology_rule_id(ontology_version: str, rule_name: str) -> UUID:
    return _stable_uuid(
        ONTOLOGY_RULE_NAMESPACE,
        ontology_version,
        rule_name,
    )


def _request_context(request: ModelingRequest) -> RequirementContext:
    return RequirementContext(
        species=request.species,
        population=request.population,
        route=request.route,
        formulation=request.formulation,
        dose=request.dose,
        dose_unit=request.dose_unit,
    )


def _gap_code(message: str) -> str:
    normalized = _normalize(message)
    if "perpetrator" in normalized:
        return "REQUEST_DDI_PERPETRATOR_MISSING"
    if "victim" in normalized:
        return "REQUEST_DDI_VICTIM_MISSING"
    if "population" in normalized:
        return "REQUEST_POPULATION_INCOMPLETE"
    if "dose regimen" in normalized:
        return "REQUEST_DOSE_REGIMEN_INCOMPLETE"
    return (
        "REQUEST_VALIDATION_GAP_"
        + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12].upper()
    )


def _identity_matches_request(
    request: ModelingRequest,
    compound: CompoundRef,
) -> bool:
    resolved_names = {
        _normalize(compound.preferred_name),
        _normalize(compound.active_moiety),
    }
    if _normalize(request.compound) not in resolved_names:
        return False
    return not request.active_moiety or (
        _normalize(request.active_moiety) in resolved_names
    )


def build_requirement_profile(
    *,
    request: ModelingRequest,
    compound_identity: CompoundRef,
    source_evidence_run_id: UUID,
    ontology_version: str,
    request_id: UUID | None = None,
    component_recommendation: RecommendationResponse | None = None,
    conflicts: list[RequirementConflictRef] | None = None,
    created_at: datetime | None = None,
) -> RequirementProfile:
    if not _identity_matches_request(request, compound_identity):
        raise RequirementProfileBuildError(
            "Resolved compound identity does not match the modeling request."
        )
    request_id = request_id or request_uuid(request)
    context = _request_context(request)
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    def add_requirement(
        *,
        category: str,
        semantic_id: str,
        preferred_name: str,
        necessity: str,
        origin: str,
        confidence: float,
        evidence_ids: Iterable[UUID] = (),
        expected_value: str | float | int | None = None,
        canonical_unit: str | None = None,
    ) -> None:
        key = (category, semantic_id)
        evidence = set(evidence_ids)
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                "category": category,
                "semantic_id": semantic_id,
                "preferred_name": preferred_name,
                "necessity": necessity,
                "origin": origin,
                "confidence": confidence,
                "evidence_ids": evidence,
                "expected_value": expected_value,
                "canonical_unit": canonical_unit,
            }
            return
        existing["evidence_ids"].update(evidence)
        existing["confidence"] = max(existing["confidence"], confidence)
        if (
            NECESSITY_PRIORITY[necessity]
            > NECESSITY_PRIORITY[existing["necessity"]]
        ):
            existing["necessity"] = necessity
        if existing["origin"] != "user_request" and origin == "user_request":
            existing["origin"] = origin

    add_requirement(
        category="compound_identity",
        semantic_id=f"compound:{compound_identity.compound_id}",
        preferred_name=compound_identity.preferred_name,
        necessity="hard_required",
        origin="user_request",
        confidence=compound_identity.identity_confidence,
        expected_value=str(compound_identity.compound_id),
    )
    add_requirement(
        category="biological_scope",
        semantic_id=f"species:{_normalize(request.species)}",
        preferred_name=request.species,
        necessity="hard_required",
        origin="user_request",
        confidence=1.0,
        expected_value=request.species,
    )
    add_requirement(
        category="validation_context",
        semantic_id=f"objective:{request.objective}",
        preferred_name=request.objective,
        necessity="hard_required",
        origin="user_request",
        confidence=1.0,
        expected_value=request.objective,
    )

    direct_values: tuple[
        tuple[str, str, Iterable[str]],
        ...,
    ] = (
        (
            "compartment",
            "compartment",
            request.required_organs,
        ),
        (
            "enzyme",
            "enzyme",
            set(request.required_enzymes) | set(request.enzyme),
        ),
        (
            "transporter",
            "transporter",
            set(request.required_transporters) | set(request.transporter),
        ),
        (
            "prediction_endpoint",
            "endpoint",
            request.prediction_endpoints,
        ),
    )
    for category, prefix, values in direct_values:
        for value in sorted(set(values), key=str.casefold):
            add_requirement(
                category=category,
                semantic_id=f"{prefix}:{_normalize(value)}",
                preferred_name=value,
                necessity="hard_required",
                origin="user_request",
                confidence=1.0,
                expected_value=value,
            )

    if request.route:
        add_requirement(
            category="administration_route",
            semantic_id=f"route:{_normalize(request.route)}",
            preferred_name=request.route,
            necessity="hard_required",
            origin="user_request",
            confidence=1.0,
            expected_value=request.route,
        )
    if request.formulation:
        add_requirement(
            category="formulation",
            semantic_id=f"formulation:{_normalize(request.formulation)}",
            preferred_name=request.formulation,
            necessity="hard_required",
            origin="user_request",
            confidence=1.0,
            expected_value=request.formulation,
        )
    for field_name, value, unit in (
        ("dose", request.dose, request.dose_unit),
        ("frequency", request.frequency, None),
        ("single_or_multiple", request.single_or_multiple, None),
        ("infusion_duration", request.infusion_duration, None),
        ("fed_or_fasted", request.fed_or_fasted, None),
        ("dosage_form", request.dosage_form, None),
        ("release_type", request.release_type, None),
    ):
        if value is None:
            continue
        add_requirement(
            category="dose_regimen",
            semantic_id=f"regimen:{field_name}:{_normalize(str(value))}",
            preferred_name=f"{field_name}: {value}",
            necessity="hard_required",
            origin="user_request",
            confidence=1.0,
            expected_value=value,
            canonical_unit=unit,
        )
    explicit_constraints: list[tuple[str, Any]] = [
        ("salt_form", request.salt_form),
        ("candidate_public_model", request.candidate_public_model),
        ("sex", request.sex),
        ("body_weight_kg", request.body_weight_kg),
        ("ethnicity", request.ethnicity),
        ("interaction_mechanism", request.interaction_mechanism),
        ("interaction_duration", request.interaction_duration),
        ("parent_or_parent_metabolite", request.parent_or_parent_metabolite),
        ("parameter_uncertainty", request.parameter_uncertainty),
        ("sensitivity_analysis", request.sensitivity_analysis),
        ("uncertainty_analysis", request.uncertainty_analysis),
        ("free_text_constraints", request.constraints),
    ]
    explicit_constraints.extend(
        ("metabolite", value) for value in request.metabolites_to_model
    )
    explicit_constraints.extend(
        ("software_preference", value)
        for value in request.software_preference
    )
    explicit_constraints.extend(
        ("calibration_dataset", value)
        for value in request.calibration_datasets
    )
    explicit_constraints.extend(
        ("validation_dataset", value)
        for value in request.validation_datasets
    )
    explicit_constraints.extend(
        ("acceptance_criterion", value)
        for value in request.acceptance_criteria
    )
    if request.victim_regimen is not None:
        explicit_constraints.append(
            ("victim_regimen", request.victim_regimen.model_dump(mode="json"))
        )
    if request.perpetrator_regimen is not None:
        explicit_constraints.append(
            (
                "perpetrator_regimen",
                request.perpetrator_regimen.model_dump(mode="json"),
            )
        )
    for field_name, value in explicit_constraints:
        if value is None or value == [] or value == "":
            continue
        add_requirement(
            category="request_constraint",
            semantic_id=(
                f"request:{field_name}:"
                + hashlib.sha256(
                    _canonical_json(value).encode("utf-8")
                ).hexdigest()[:16]
            ),
            preferred_name=f"{field_name}: {value}",
            necessity="hard_required",
            origin="user_request",
            confidence=1.0,
            expected_value=(
                value if isinstance(value, (str, float, int)) else _canonical_json(value)
            ),
        )
    if request.whole_body_or_reduced_pbpk not in {None, "unspecified"}:
        add_requirement(
            category="model_structure",
            semantic_id=(
                "model_structure:"
                + request.whole_body_or_reduced_pbpk
            ),
            preferred_name=request.whole_body_or_reduced_pbpk,
            necessity="hard_required",
            origin="user_request",
            confidence=1.0,
            expected_value=request.whole_body_or_reduced_pbpk,
        )
    for role, value in (
        ("victim", request.victim_drug),
        ("perpetrator", request.perpetrator_drug),
    ):
        if value:
            add_requirement(
                category="ddi_role",
                semantic_id=f"ddi:{role}:{_normalize(value)}",
                preferred_name=f"{role}: {value}",
                necessity="hard_required",
                origin="user_request",
                confidence=1.0,
                expected_value=value,
            )
    for field_name, value in (
        ("population", request.population),
        ("age_range", request.age_range),
        ("renal_function", request.renal_function),
        ("hepatic_function", request.hepatic_function),
        ("disease_state", request.disease_state),
        ("genotype", request.genotype),
        ("pregnancy_status", request.pregnancy_status),
    ):
        if value:
            add_requirement(
                category="population_feature",
                semantic_id=(
                    f"population:{field_name}:{_normalize(value)}"
                ),
                preferred_name=f"{field_name}: {value}",
                necessity="hard_required",
                origin="user_request",
                confidence=1.0,
                expected_value=value,
            )

    for process in OBJECTIVE_PROCESS_RULES[request.objective]:
        rule_name = f"objective:{request.objective}:requires:{process}"
        add_requirement(
            category="adme_process",
            semantic_id=f"process:{process}",
            preferred_name=process,
            necessity="hard_required",
            origin="ontology_rule",
            confidence=1.0,
            evidence_ids=[_ontology_rule_id(ontology_version, rule_name)],
            expected_value=process,
        )

    evidence_by_semantic: dict[tuple[str, str], set[UUID]] = {}
    if component_recommendation is not None:
        if (
            _normalize(component_recommendation.compound_name)
            not in {
                _normalize(compound_identity.preferred_name),
                _normalize(compound_identity.active_moiety),
            }
        ):
            raise RequirementProfileBuildError(
                "Component recommendation compound does not match profile."
            )
        for component in component_recommendation.components:
            mapping = COMPONENT_TO_REQUIREMENT.get(component.component)
            if mapping is None or component.status == "insufficient_evidence":
                continue
            category, semantic_name = mapping
            evidence_ids = {
                _validated_uuid(
                    citation.evidence_id,
                    field="citation.evidence_id",
                )
                for citation in component.citations
            }
            if not evidence_ids:
                continue
            semantic_id = (
                f"process:{semantic_name}"
                if category == "adme_process"
                else f"{category}:{semantic_name}"
            )
            evidence_by_semantic.setdefault(
                (category, semantic_id),
                set(),
            ).update(evidence_ids)
            add_requirement(
                category=category,
                semantic_id=semantic_id,
                preferred_name=semantic_name,
                necessity=(
                    "preferred"
                    if component.status == "recommended"
                    else "optional"
                ),
                origin="validated_evidence",
                confidence=component.confidence,
                evidence_ids=evidence_ids,
                expected_value=semantic_name,
            )

    requirements: list[RequirementItem] = []
    for key, value in sorted(merged.items()):
        evidence_ids = sorted(value.pop("evidence_ids"), key=str)
        value["evidence_ids"] = evidence_ids
        requirements.append(
            RequirementItem(
                requirement_id=_stable_uuid(
                    REQUIREMENT_NAMESPACE,
                    str(request_id),
                    key,
                    context.model_dump(mode="json"),
                ),
                context=context,
                **value,
            )
        )

    requirement_by_semantic = {
        (item.category, item.semantic_id): item for item in requirements
    }
    gaps: list[RequirementGap] = [
        RequirementGap(
            gap_code=_gap_code(message),
            severity="critical",
            requirement_ids=[],
            blocking=True,
            source="request_validation",
        )
        for message in request.request_gaps()
    ]
    for process in OBJECTIVE_PROCESS_RULES[request.objective]:
        semantic_id = f"process:{process}"
        requirement = requirement_by_semantic[
            ("adme_process", semantic_id)
        ]
        if request.objective == "public_model_reuse":
            continue
        if evidence_by_semantic.get(("adme_process", semantic_id)):
            continue
        gaps.append(
            RequirementGap(
                gap_code=(
                    "HARD_REQUIREMENT_EVIDENCE_MISSING_"
                    + process.upper()
                ),
                severity="critical",
                requirement_ids=[requirement.requirement_id],
                blocking=True,
                source="evidence_coverage",
            )
        )

    conflicts = conflicts or []
    eligibility = (
        "blocked"
        if gaps or any(conflict.severity == "hard" for conflict in conflicts)
        else "ready"
    )
    created_at = created_at or datetime.now(UTC)
    profile_id = _stable_uuid(
        PROFILE_NAMESPACE,
        str(request_id),
        str(source_evidence_run_id),
        ontology_version,
        PROFILE_VERSION,
    )
    payload: dict[str, Any] = {
        "profile_id": profile_id,
        "profile_version": PROFILE_VERSION,
        "request_id": request_id,
        "ontology_version": ontology_version,
        "compound_identity": compound_identity,
        "requirements": requirements,
        "critical_gaps": gaps,
        "conflicts": conflicts,
        "source_evidence_run_id": source_evidence_run_id,
        "ranking_eligibility": eligibility,
        "created_at": created_at,
    }
    draft = RequirementProfile.model_construct(
        checksum="0" * 64,
        **payload,
    )
    payload["checksum"] = requirement_profile_checksum(
        draft.model_dump(mode="json")
    )
    return RequirementProfile.model_validate(payload)


def audit_requirement_profiles(
    profiles: Iterable[RequirementProfile],
    *,
    known_provenance_ids: set[UUID] | None = None,
) -> dict[str, Any]:
    profile_list = list(profiles)
    profile_ids = [profile.profile_id for profile in profile_list]
    request_ids = [profile.request_id for profile in profile_list]
    requirement_ids = [
        requirement.requirement_id
        for profile in profile_list
        for requirement in profile.requirements
    ]
    pydantic_validation_errors: list[str] = []
    for profile in profile_list:
        try:
            RequirementProfile.model_validate(
                profile.model_dump(mode="json")
            )
        except Exception as exc:
            pydantic_validation_errors.append(
                f"{profile.profile_id}:{type(exc).__name__}:{exc}"
            )
    hard_requirements = [
        requirement
        for profile in profile_list
        for requirement in profile.requirements
        if requirement.necessity == "hard_required"
    ]
    hard_provenance_errors = [
        str(requirement.requirement_id)
        for requirement in hard_requirements
        if requirement.origin != "user_request"
        and not requirement.evidence_ids
    ]
    unknown_provenance_errors: list[str] = []
    if known_provenance_ids is not None:
        for profile in profile_list:
            for requirement in profile.requirements:
                if requirement.origin == "user_request":
                    continue
                unknown_ids = sorted(
                    set(requirement.evidence_ids)
                    - known_provenance_ids,
                    key=str,
                )
                unknown_provenance_errors.extend(
                    f"{requirement.requirement_id}:{value}"
                    for value in unknown_ids
                )
    conflict_blocking_errors = [
        str(profile.profile_id)
        for profile in profile_list
        if any(
            conflict.severity == "hard"
            for conflict in profile.conflicts
        )
        and profile.ranking_eligibility != "blocked"
    ]
    checks = {
        "profiles_present": bool(profile_list),
        "profile_ids_unique": len(profile_ids) == len(set(profile_ids)),
        "request_ids_unique": len(request_ids) == len(set(request_ids)),
        "requirement_ids_unique": (
            len(requirement_ids) == len(set(requirement_ids))
        ),
        "pydantic_valid_rate_1": not pydantic_validation_errors,
        "hard_requirement_provenance_complete": (
            not hard_provenance_errors
        ),
        "provenance_references_resolve": (
            not unknown_provenance_errors
        ),
        "unresolved_compound_identity_count_zero": all(
            profile.compound_identity.identity_confidence > 0
            and bool(profile.compound_identity.preferred_name.strip())
            and bool(profile.compound_identity.active_moiety.strip())
            for profile in profile_list
        ),
        "hard_conflict_blocks_ranking": not conflict_blocking_errors,
    }
    return {
        "audit_schema_version": "1.0.0",
        "builder_version": PROFILE_BUILDER_VERSION,
        "audited_at": datetime.now(UTC).isoformat(),
        "accepted": all(checks.values()),
        "checks": checks,
        "counts": {
            "profiles": len(profile_list),
            "requirements": sum(
                len(profile.requirements) for profile in profile_list
            ),
            "hard_requirements": len(hard_requirements),
            "blocked_profiles": sum(
                profile.ranking_eligibility == "blocked"
                for profile in profile_list
            ),
        },
        "errors": {
            "pydantic_validation": pydantic_validation_errors,
            "hard_requirement_provenance": hard_provenance_errors,
            "unknown_provenance_references": (
                unknown_provenance_errors
            ),
            "hard_conflict_not_blocking": conflict_blocking_errors,
        },
    }


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RequirementProfileBuildError(
                    f"Expected object at {path}:{line_number}."
                )
            yield value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row))
            handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and audit strong-typed RequirementProfiles."
    )
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ontology-version", required=True)
    args = parser.parse_args()

    profiles: list[RequirementProfile] = []
    profile_index: list[dict[str, Any]] = []
    provenance_by_id: dict[str, dict[str, Any]] = {}
    for bundle in _iter_jsonl(args.bundles):
        request = ModelingRequest.model_validate(
            bundle["modeling_request"]
        )
        recommendation_payload = bundle.get("component_recommendation")
        recommendation = (
            RecommendationResponse.model_validate(
                recommendation_payload
            )
            if recommendation_payload
            else None
        )
        profile = build_requirement_profile(
            request=request,
            request_id=(
                UUID(str(bundle["request_id"]))
                if bundle.get("request_id")
                else None
            ),
            compound_identity=CompoundRef.model_validate(
                bundle["compound_identity"]
            ),
            source_evidence_run_id=UUID(
                str(bundle["source_evidence_run_id"])
            ),
            ontology_version=args.ontology_version,
            component_recommendation=recommendation,
            conflicts=[
                RequirementConflictRef.model_validate(item)
                for item in bundle.get("conflicts") or []
            ],
        )
        profiles.append(profile)
        profile_index.append(
            {
                "profile_id": str(profile.profile_id),
                "request_id": str(profile.request_id),
                "source_example_id": bundle.get("source_example_id"),
                "benchmark_level": bundle.get("benchmark_level"),
                "benchmark_split": bundle.get("benchmark_split"),
                "compound_id": str(
                    profile.compound_identity.compound_id
                ),
                "objective": request.objective,
            }
        )
        if recommendation is not None:
            for component in recommendation.components:
                for citation in component.citations:
                    provenance_by_id.setdefault(
                        citation.evidence_id,
                        {
                            "provenance_id": citation.evidence_id,
                            "provenance_kind": "validated_evidence",
                            "source_id": citation.source_id,
                            "source_location": citation.source_location,
                            "source_text_sha256": hashlib.sha256(
                                citation.source_text.encode("utf-8")
                            ).hexdigest(),
                        },
                    )
        for process in OBJECTIVE_PROCESS_RULES[request.objective]:
            rule_name = (
                f"objective:{request.objective}:requires:{process}"
            )
            rule_id = str(
                _ontology_rule_id(
                    args.ontology_version,
                    rule_name,
                )
            )
            provenance_by_id.setdefault(
                rule_id,
                {
                    "provenance_id": rule_id,
                    "provenance_kind": "ontology_rule",
                    "ontology_version": args.ontology_version,
                    "rule_name": rule_name,
                },
            )

    args.output.mkdir(parents=True, exist_ok=False)
    _atomic_jsonl(
        args.output / "requirement_profiles.jsonl",
        [
            profile.model_dump(mode="json")
            for profile in profiles
        ],
    )
    _atomic_jsonl(
        args.output / "profile_index.jsonl",
        profile_index,
    )
    _atomic_jsonl(
        args.output / "provenance_catalog.jsonl",
        [
            provenance_by_id[key]
            for key in sorted(provenance_by_id)
        ],
    )
    _atomic_json(
        args.output / "requirement_profile.schema.json",
        RequirementProfile.model_json_schema(),
    )
    audit = audit_requirement_profiles(
        profiles,
        known_provenance_ids={
            UUID(value) for value in provenance_by_id
        },
    )
    _atomic_json(args.output / "audit.json", audit)
    if not audit["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
