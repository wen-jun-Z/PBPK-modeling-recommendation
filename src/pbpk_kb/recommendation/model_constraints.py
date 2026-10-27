from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from uuid import UUID, uuid5

from pbpk_kb.model_assets.models import (
    ExecutionProbeStatus,
    LicenseStatus,
    OSPModelVersionSnapshot,
    ParseStatus,
)
from pbpk_kb.recommendation.schemas import (
    ConstraintCheck,
    DecisionReasonCode,
    RequirementItem,
    RequirementProfile,
)
from pbpk_kb.recommendation.transferability import (
    assess_numeric_transferability,
)


MODEL_VERSION_NAMESPACE = UUID("ba57a7e9-3ff4-4a42-81c3-924f14ffbdce")
CONSTRAINT_EVALUATOR_VERSION = "pbpk-model-constraints-1.3.0"


def model_version_uuid(source_model_version_key: str) -> UUID:
    return uuid5(MODEL_VERSION_NAMESPACE, source_model_version_key)


def _normalize(value: str) -> str:
    return " ".join(
        re.findall(r"[a-z0-9]+", value.casefold())
    )


def _normalized_set(values: Iterable[str]) -> set[str]:
    return {
        normalized
        for value in values
        if (normalized := _normalize(value))
    }


def _contains(
    requested: str,
    supported: Iterable[str],
) -> bool:
    target = _normalize(requested)
    if not target:
        return False
    values = _normalized_set(supported)
    return target in values or any(
        target in value or value in target
        for value in values
        if len(value) >= 3
    )


def _asset_locations(model: OSPModelVersionSnapshot) -> list[str]:
    locations = sorted(
        {
            asset.source_location
            for asset in model.assets
            if asset.source_location
        }
    )
    return locations or [
        f"MODEL_VERSION_SNAPSHOT:{model.model_version_id}"
    ]


def _check(
    *,
    model: OSPModelVersionSnapshot,
    requirement: RequirementItem,
    constraint_name: str,
    severity: str,
    result: str,
    observed_value: Any,
    reason_code: DecisionReasonCode,
    include_requirement_evidence: bool = True,
) -> ConstraintCheck:
    return ConstraintCheck(
        constraint_id=(
            f"{CONSTRAINT_EVALUATOR_VERSION}:"
            f"{constraint_name}:{requirement.requirement_id}"
        ),
        model_version_id=model_version_uuid(model.model_version_id),
        source_model_version_key=model.model_version_id,
        requirement_id=requirement.requirement_id,
        severity=severity,
        result=result,
        observed_value=observed_value,
        reason_code=reason_code,
        evidence_ids=(
            requirement.evidence_ids
            if include_requirement_evidence
            else []
        ),
        model_asset_locations=_asset_locations(model),
    )


def _pass_or_fail(
    *,
    model: OSPModelVersionSnapshot,
    requirement: RequirementItem,
    constraint_name: str,
    passed: bool,
    observed_value: Any,
    failure_reason: DecisionReasonCode,
    severity: str | None = None,
) -> ConstraintCheck:
    return _check(
        model=model,
        requirement=requirement,
        constraint_name=constraint_name,
        severity=severity
        or (
            "hard"
            if requirement.necessity == "hard_required"
            else "soft"
        ),
        result="pass" if passed else "fail",
        observed_value=observed_value,
        reason_code=(
            "REQUIREMENT_SATISFIED"
            if passed
            else failure_reason
        ),
    )


def _anchor_requirement(
    profile: RequirementProfile,
) -> RequirementItem:
    return next(
        (
            item
            for item in profile.requirements
            if item.category == "validation_context"
        ),
        profile.requirements[0],
    )


def _validation_scope_values(
    profile: RequirementProfile,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "species": None,
        "route": None,
        "formulation": None,
        "dose": None,
        "dose_unit": None,
        "frequency": None,
        "single_or_multiple": None,
        "population": None,
        "endpoints": [],
    }
    for requirement in profile.requirements:
        expected = requirement.expected_value
        if requirement.category == "biological_scope":
            values["species"] = str(expected or requirement.preferred_name)
        elif requirement.category == "administration_route":
            values["route"] = str(expected or requirement.preferred_name)
        elif requirement.category == "formulation":
            values["formulation"] = str(expected or requirement.preferred_name)
        elif requirement.category == "population_feature":
            values["population"] = str(expected or requirement.preferred_name)
        elif requirement.category == "prediction_endpoint":
            values["endpoints"].append(
                str(expected or requirement.preferred_name)
            )
        elif requirement.category == "dose_regimen":
            if requirement.semantic_id.startswith("regimen:dose:"):
                values["dose"] = (
                    float(expected) if expected is not None else None
                )
                values["dose_unit"] = requirement.canonical_unit
            elif requirement.semantic_id.startswith("regimen:frequency:"):
                values["frequency"] = str(expected)
            elif requirement.semantic_id.startswith(
                "regimen:single_or_multiple:"
            ):
                values["single_or_multiple"] = str(expected)
    values["endpoints"] = sorted(set(values["endpoints"]))
    return values


def _frequency_matches(requested: str, observed: str | None) -> bool:
    aliases = {
        "qd": "once daily",
        "once per day": "once daily",
        "once daily": "once daily",
        "single dose": "single dose",
        "bid": "twice daily",
        "twice daily": "twice daily",
        "tid": "three times daily",
        "three times daily": "three times daily",
    }
    target = aliases.get(_normalize(requested), _normalize(requested))
    value = aliases.get(_normalize(observed or ""), _normalize(observed or ""))
    return bool(target and value and target == value)


def _validation_scope_check(
    profile: RequirementProfile,
    model: OSPModelVersionSnapshot,
) -> ConstraintCheck:
    anchor = _anchor_requirement(profile)
    scope = model.validation_scope
    requested = _validation_scope_values(profile)
    objective = _normalize(
        str(anchor.expected_value or anchor.preferred_name)
    )
    missing_scope_fields = [
        field
        for field, present in (
            ("route", bool(requested["route"])),
            ("dose", requested["dose"] is not None),
            ("dose_unit", bool(requested["dose_unit"])),
            ("frequency", bool(requested["frequency"])),
            (
                "single_or_multiple",
                bool(requested["single_or_multiple"]),
            ),
            ("formulation", bool(requested["formulation"])),
            ("prediction_endpoints", bool(requested["endpoints"])),
        )
        if not present
    ]
    if objective == "public model reuse" and missing_scope_fields:
        return _check(
            model=model,
            requirement=anchor,
            constraint_name="validation_scope_transferability",
            severity="hard",
            result="unknown",
            observed_value={
                "requested": requested,
                "missing_required_scope_fields": missing_scope_fields,
                "reason": (
                    "Direct public-model reuse requires a complete explicit "
                    "administration contract and at least one prediction "
                    "endpoint before validation-scope transferability can be "
                    "assessed. Missing values are not wildcards."
                ),
            },
            reason_code="VALIDATION_NOT_TRANSFERABLE",
        )
    if scope is None:
        return _check(
            model=model,
            requirement=anchor,
            constraint_name="validation_scope_transferability",
            severity="hard",
            result="unknown",
            observed_value={
                "requested": requested,
                "reason": "No validation-bound scenario scope is registered.",
            },
            reason_code="VALIDATION_NOT_TRANSFERABLE",
        )

    def scenario_context_matches(scenario: Any) -> bool:
        if requested["species"] and not _contains(
            requested["species"], scenario.species
        ):
            return False
        if requested["route"] and not _contains(
            requested["route"], scenario.routes
        ):
            return False
        if requested["formulation"] and not _contains(
            requested["formulation"], scenario.formulations
        ):
            return False
        if requested["population"] and not _contains(
            requested["population"], scenario.populations
        ):
            return False
        if requested["endpoints"] and any(
            not _contains(endpoint, scenario.evaluated_endpoints)
            for endpoint in requested["endpoints"]
        ):
            return False
        if requested["frequency"] and not _frequency_matches(
            requested["frequency"], scenario.frequency
        ):
            return False
        if (
            requested["single_or_multiple"]
            and requested["single_or_multiple"]
            != scenario.single_or_multiple
        ):
            return False
        return True

    context_matching = [
        scenario
        for scenario in scope.scenarios
        if scenario_context_matches(scenario)
    ]
    numeric_transferability = None
    matching = context_matching
    if requested["dose"] is not None:
        validated_doses = [
            (float(scenario.dose), str(scenario.dose_unit))
            for scenario in context_matching
            if scenario.dose is not None and scenario.dose_unit
        ]
        if requested["dose_unit"]:
            numeric_transferability = assess_numeric_transferability(
                requested_value=float(requested["dose"]),
                requested_unit=str(requested["dose_unit"]),
                validated_values=validated_doses,
            )
        matching = [
            scenario
            for scenario in context_matching
            if scenario.dose is not None
            and scenario.dose_unit
            and requested["dose_unit"]
            and assess_numeric_transferability(
                requested_value=float(requested["dose"]),
                requested_unit=str(requested["dose_unit"]),
                validated_values=[
                    (float(scenario.dose), str(scenario.dose_unit))
                ],
            ).direct_reuse_equivalent
        ]
    return _pass_or_fail(
        model=model,
        requirement=anchor,
        constraint_name="validation_scope_transferability",
        passed=bool(matching),
        observed_value={
            "requested": requested,
            "matching_scenario_ids": [
                scenario.simulation_id for scenario in matching
            ],
            "context_matching_scenario_ids": [
                scenario.simulation_id for scenario in context_matching
            ],
            "numeric_transferability": (
                numeric_transferability.to_dict()
                if numeric_transferability is not None
                else None
            ),
            "transferability_semantics": (
                "Exact and numerically equivalent continuous inputs may "
                "reuse a registered scenario. Interpolation and extrapolation "
                "remain ranked adaptation candidates and require rerunning "
                "the simulation; they do not make the model irrelevant."
            ),
            "registered_scenario_count": len(scope.scenarios),
            "validation_scope": {
                "routes": scope.routes,
                "formulations": scope.formulations,
                "evaluated_endpoints": scope.evaluated_endpoints,
                "dataset_count": len(scope.dataset_names),
            },
        },
        failure_reason="VALIDATION_NOT_TRANSFERABLE",
        severity="hard",
    )


def _asset_checks(
    profile: RequirementProfile,
    model: OSPModelVersionSnapshot,
) -> list[ConstraintCheck]:
    anchor = _anchor_requirement(profile)
    project_asset = next(
        (
            asset
            for asset in model.assets
            if asset.asset_kind in {"pksim_project", "pksim_snapshot"}
            and asset.relative_path == model.project_relative_path
        ),
        None,
    )
    asset_present = project_asset is not None
    checksum_matches = (
        project_asset is not None
        and project_asset.sha256 == model.project_sha256
    )
    parser_passed = (
        model.static_parse_status
        in {
            ParseStatus.PARSED,
            ParseStatus.REGISTERED_NOT_EXECUTABLE,
        }
        and model.pkml_parse_status == ParseStatus.PARSED
    )
    execution_passed = (
        model.execution_probe_status == ExecutionProbeStatus.PASSED
    )
    license_allowed = model.license.status == LicenseStatus.ALLOWED
    validation_label = model.validation_run_label
    validation_passed = bool(
        validation_label is not None
        and validation_label.validation_grade == "external_study_validated"
        and validation_label.run_label_grade == 3
        and validation_label.ranking_label == "reuse_positive"
        and validation_label.engineering_accepted
        and validation_label.scientific_accepted
        and validation_label.reusable_candidate
    )
    return [
        _pass_or_fail(
            model=model,
            requirement=anchor,
            constraint_name="asset_present",
            passed=asset_present,
            observed_value=model.project_relative_path,
            failure_reason="ASSET_MISSING",
            severity="hard",
        ),
        _pass_or_fail(
            model=model,
            requirement=anchor,
            constraint_name="project_checksum",
            passed=checksum_matches,
            observed_value=model.project_sha256,
            failure_reason="CHECKSUM_FAILED",
            severity="hard",
        ),
        _pass_or_fail(
            model=model,
            requirement=anchor,
            constraint_name="license",
            passed=license_allowed,
            observed_value=model.license.status.value,
            failure_reason="LICENSE_UNRESOLVED",
            severity="hard",
        ),
        _pass_or_fail(
            model=model,
            requirement=anchor,
            constraint_name="parser",
            passed=parser_passed,
            observed_value={
                "static_project_parser": (
                    model.static_parse_status.value
                ),
                "exported_pkml_parser": (
                    model.pkml_parse_status.value
                ),
            },
            failure_reason="PARSER_FAILED",
            severity="hard",
        ),
        _pass_or_fail(
            model=model,
            requirement=anchor,
            constraint_name="execution_probe",
            passed=execution_passed,
            observed_value=model.execution_probe_status.value,
            failure_reason="EXECUTION_PROBE_FAILED",
            severity="hard",
        ),
        _pass_or_fail(
            model=model,
            requirement=anchor,
            constraint_name="external_validation_run",
            passed=validation_passed,
            observed_value=(
                validation_label.model_dump(mode="json")
                if validation_label is not None
                else None
            ),
            failure_reason=(
                "VALIDATION_CRITERIA_MISSING"
                if validation_label is None
                else "VALIDATION_FAILED"
            ),
            severity="hard",
        ),
    ]


def _profile_blocker_checks(
    profile: RequirementProfile,
    model: OSPModelVersionSnapshot,
) -> list[ConstraintCheck]:
    anchor = _anchor_requirement(profile)
    requirement_by_id = {
        requirement.requirement_id: requirement
        for requirement in profile.requirements
    }
    checks: list[ConstraintCheck] = []
    for gap in profile.critical_gaps:
        if not gap.blocking:
            continue
        requirement_ids = gap.requirement_ids or [
            anchor.requirement_id
        ]
        for requirement_id in requirement_ids:
            requirement = requirement_by_id[requirement_id]
            checks.append(
                _check(
                    model=model,
                    requirement=requirement,
                    constraint_name=f"profile_gap:{gap.gap_code}",
                    severity="hard",
                    result="fail",
                    observed_value=gap.gap_code,
                    reason_code="CRITICAL_EVIDENCE_GAP",
                )
            )
    for conflict in profile.conflicts:
        if conflict.severity != "hard":
            continue
        for requirement_id in conflict.requirement_ids:
            requirement = requirement_by_id[requirement_id]
            checks.append(
                _check(
                    model=model,
                    requirement=requirement,
                    constraint_name=(
                        f"profile_conflict:{conflict.conflict_id}"
                    ),
                    severity="hard",
                    result="fail",
                    observed_value=conflict.reason_code,
                    reason_code="UNRESOLVED_CONFLICT",
                )
            )
    return checks


def _adme_process_supported(
    process: str,
    model: OSPModelVersionSnapshot,
) -> bool:
    process = _normalize(process)
    process_types = _normalized_set(model.process_types)
    if process == "absorption":
        return "absorption" in process_types
    if process == "distribution":
        return (
            "distribution" in process_types
            or len(model.compartments) >= 5
        )
    if process == "metabolism":
        return "metabolism" in process_types or bool(model.enzymes)
    if process == "transport":
        return "transport" in process_types or bool(model.transporters)
    if process == "excretion":
        return bool(
            {
                "renal excretion",
                "biliary excretion",
                "excretion",
            }
            & process_types
        )
    if process == "renal excretion":
        return "renal excretion" in process_types
    if process == "ddi":
        return (
            len(model.compounds) >= 2
            and bool(model.enzymes or model.transporters)
        )
    return False


def _endpoint_supported(
    endpoint: str,
    model: OSPModelVersionSnapshot,
) -> bool:
    target = _normalize(endpoint)
    validated_endpoints = _normalized_set(
        model.validation_scope.evaluated_endpoints
        if model.validation_scope is not None
        else []
    )
    if target in validated_endpoints:
        return True
    output_names = _normalized_set(model.output_names)
    endpoint_patterns: dict[str, tuple[str, ...]] = {
        "auc": ("auc",),
        "cmax": ("cmax", "maximum concentration"),
        "tmax": ("tmax", "time to maximum"),
        "half life": ("half life", "half-life"),
        "clearance": ("clearance",),
        "volume of distribution": (
            "volume of distribution",
            "vss",
        ),
        "concentration time profile": (
            "concentration",
            "plasma",
            "blood",
        ),
        "tissue concentration": ("tissue", "concentration"),
        "fraction absorbed": ("fraction absorbed",),
        "bioavailability": ("bioavailability",),
        "renal excretion": ("renal", "urine", "excretion"),
        "ddi auc ratio": ("auc ratio", "ddi"),
        "ddi cmax ratio": ("cmax ratio", "ddi"),
    }
    patterns = endpoint_patterns.get(target, (target,))
    return any(
        _normalize(pattern) in output
        for pattern in patterns
        for output in output_names
    )


def _requirement_check(
    *,
    profile: RequirementProfile,
    requirement: RequirementItem,
    model: OSPModelVersionSnapshot,
) -> ConstraintCheck:
    expected = requirement.expected_value
    expected_text = (
        str(expected)
        if expected is not None
        else requirement.preferred_name
    )
    category = requirement.category
    if category == "compound_identity":
        active_names = {
            profile.compound_identity.preferred_name,
            profile.compound_identity.active_moiety,
        }
        passed = any(
            _contains(name, model.compounds)
            or _normalize(name) == _normalize(model.model_family)
            for name in active_names
        )
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="compound_identity",
            passed=passed,
            observed_value=model.compounds,
            failure_reason="COMPOUND_MISMATCH",
        )
    if category == "biological_scope":
        passed = _contains(expected_text, model.species)
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="species",
            passed=passed,
            observed_value=model.species,
            failure_reason="SPECIES_MISMATCH",
        )
    if category == "administration_route":
        passed = _contains(expected_text, model.routes)
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="route",
            passed=passed,
            observed_value=model.routes,
            failure_reason="ROUTE_UNSUPPORTED",
        )
    if category == "formulation":
        passed = _contains(expected_text, model.formulations)
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="formulation",
            passed=passed,
            observed_value=model.formulations,
            failure_reason="FORMULATION_UNSUPPORTED",
        )
    if category == "dose_regimen":
        scope_check = _validation_scope_check(profile, model)
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="dose_regimen_scope",
            passed=scope_check.result == "pass",
            observed_value=scope_check.observed_value,
            failure_reason="VALIDATION_NOT_TRANSFERABLE",
            severity="hard",
        )
    if category == "request_constraint":
        return _check(
            model=model,
            requirement=requirement,
            constraint_name="explicit_request_constraint",
            severity="hard",
            result="unknown",
            observed_value={
                "requested": expected,
                "reason": (
                    "This explicit request field is not represented in the "
                    "registered model applicability contract."
                ),
            },
            reason_code="VALIDATION_NOT_TRANSFERABLE",
        )
    if category == "enzyme":
        passed = _contains(expected_text, model.enzymes)
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="enzyme",
            passed=passed,
            observed_value=model.enzymes,
            failure_reason="REQUIRED_MECHANISM_MISSING",
        )
    if category == "transporter":
        passed = _contains(expected_text, model.transporters)
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="transporter",
            passed=passed,
            observed_value=model.transporters,
            failure_reason="REQUIRED_MECHANISM_MISSING",
        )
    if category == "compartment":
        passed = _contains(expected_text, model.compartments)
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="compartment",
            passed=passed,
            observed_value=model.compartments,
            failure_reason="REQUIRED_MECHANISM_MISSING",
        )
    if category == "adme_process":
        passed = _adme_process_supported(expected_text, model)
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="adme_process",
            passed=passed,
            observed_value=model.process_types,
            failure_reason="REQUIRED_MECHANISM_MISSING",
        )
    if category == "prediction_endpoint":
        passed = _endpoint_supported(expected_text, model)
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="prediction_endpoint",
            passed=passed,
            observed_value=model.output_names,
            failure_reason="ENDPOINT_UNSUPPORTED",
        )
    if category == "population_feature":
        passed = _contains(expected_text, model.populations)
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="population",
            passed=passed,
            observed_value=model.populations,
            failure_reason="VALIDATION_NOT_TRANSFERABLE",
        )
    if category == "ddi_role":
        passed = _contains(expected_text, model.compounds)
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="ddi_role",
            passed=passed,
            observed_value=model.compounds,
            failure_reason="REQUIRED_MECHANISM_MISSING",
        )
    if category == "model_structure":
        normalized = _normalize(expected_text)
        if normalized == "whole body":
            passed = len(model.compartments) >= 10
        elif normalized == "reduced":
            passed = 1 <= len(model.compartments) < 10
        elif normalized == "public model template":
            passed = True
        else:
            passed = False
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="model_structure",
            passed=passed,
            observed_value={
                "compartment_count": len(model.compartments),
                "expected": expected_text,
            },
            failure_reason="REQUIRED_MECHANISM_MISSING",
        )
    if category == "validation_context":
        supported = _normalize(expected_text) in {
            "public model reuse",
            "base pk",
        }
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="validation_objective",
            passed=supported,
            observed_value={
                "requested_objective": expected_text,
                "validated_objectives": ["public_model_reuse", "base_pk"],
            },
            failure_reason="VALIDATION_NOT_TRANSFERABLE",
            severity="hard",
        )
    if category == "parameter":
        passed = model.parameter_record_count > 0
        return _pass_or_fail(
            model=model,
            requirement=requirement,
            constraint_name="parameter_coverage",
            passed=passed,
            observed_value=model.parameter_record_count,
            failure_reason="REQUIRED_MECHANISM_MISSING",
        )
    return _check(
        model=model,
        requirement=requirement,
        constraint_name=f"{category}_not_applicable",
        severity=(
            "hard"
            if requirement.necessity == "hard_required"
            else "soft"
        ),
        result="not_applicable",
        observed_value=None,
        reason_code="NOT_APPLICABLE",
        include_requirement_evidence=False,
    )


def evaluate_model_constraints(
    *,
    profile: RequirementProfile,
    model: OSPModelVersionSnapshot,
) -> list[ConstraintCheck]:
    checks = _asset_checks(profile, model)
    checks.append(_validation_scope_check(profile, model))
    checks.extend(_profile_blocker_checks(profile, model))
    checks.extend(
        _requirement_check(
            profile=profile,
            requirement=requirement,
            model=model,
        )
        for requirement in profile.requirements
    )
    constraint_ids = [check.constraint_id for check in checks]
    if len(constraint_ids) != len(set(constraint_ids)):
        raise ValueError(
            f"Duplicate constraint IDs for {model.model_version_id}."
        )
    return checks
