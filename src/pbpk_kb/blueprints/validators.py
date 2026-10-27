from __future__ import annotations

import math
from datetime import UTC, datetime

from pbpk_kb.blueprints.schemas import (
    BlueprintTemplate,
    BlueprintValidationReport,
    DDIProcess,
    EstimatedParameter,
    MetabolismProcess,
    MissingParameter,
    PBPKBlueprint,
    TransportProcess,
)
from pbpk_kb.blueprints.template_catalog import (
    required_process_type_options,
)
from pbpk_kb.recommendation.schemas import RequirementProfile


VALIDATOR_VERSION = "pbpk-blueprint-validator-1.0.0"


def _append(
    errors: dict[str, list[dict[str, object]]],
    category: str,
    **payload: object,
) -> None:
    errors[category].append(payload)


def validate_blueprint(
    *,
    blueprint: PBPKBlueprint,
    template: BlueprintTemplate,
    profile: RequirementProfile,
) -> BlueprintValidationReport:
    error_categories = [
        "profile",
        "template",
        "requirements",
        "compartments",
        "processes",
        "mass_balance",
        "dimensions",
        "units",
        "parameters",
        "provenance",
        "population",
        "dosing",
        "datasets",
    ]
    errors: dict[str, list[dict[str, object]]] = {
        category: [] for category in error_categories
    }

    if blueprint.profile_id != profile.profile_id:
        _append(errors, "profile", reason="PROFILE_ID_MISMATCH")
    if blueprint.profile_checksum != profile.checksum:
        _append(errors, "profile", reason="PROFILE_CHECKSUM_MISMATCH")
    if profile.ranking_eligibility != "ready":
        _append(errors, "profile", reason="PROFILE_NOT_READY")
    if (
        blueprint.compound_semantic_id
        != f"compound:{profile.compound_identity.compound_id}"
    ):
        _append(errors, "profile", reason="COMPOUND_IDENTITY_MISMATCH")
    if blueprint.compound_name.casefold() != (
        profile.compound_identity.preferred_name.casefold()
    ):
        _append(errors, "profile", reason="COMPOUND_NAME_MISMATCH")

    if blueprint.template.template_id != template.template_id:
        _append(errors, "template", reason="TEMPLATE_ID_MISMATCH")
    if blueprint.template.skeleton_sha256 != (
        template.source_skeleton_sha256
    ):
        _append(errors, "template", reason="SKELETON_CHECKSUM_MISMATCH")
    if blueprint.template.license_sha256 != template.license_sha256:
        _append(errors, "template", reason="LICENSE_CHECKSUM_MISMATCH")
    if blueprint.target_runtime != template.target_runtime:
        _append(errors, "template", reason="TARGET_RUNTIME_MISMATCH")
    if (
        blueprint.compound_semantic_id
        not in template.compound_semantic_ids
    ):
        _append(
            errors,
            "template",
            reason="TEMPLATE_COMPOUND_UNSUPPORTED",
        )
    if blueprint.species.casefold() != template.species.casefold():
        _append(errors, "template", reason="TEMPLATE_SPECIES_UNSUPPORTED")
    if blueprint.objective.casefold() not in set(
        map(str.casefold, template.supported_objectives)
    ):
        _append(errors, "template", reason="TEMPLATE_OBJECTIVE_UNSUPPORTED")

    hard_requirement_ids = {
        item.requirement_id
        for item in profile.requirements
        if item.necessity == "hard_required"
    }
    satisfied = set(blueprint.satisfied_requirement_ids)
    unresolved = set(blueprint.unresolved_requirement_ids)
    missing_hard = hard_requirement_ids - satisfied
    if missing_hard:
        _append(
            errors,
            "requirements",
            reason="HARD_REQUIREMENTS_NOT_SATISFIED",
            requirement_ids=sorted(map(str, missing_hard)),
        )
    if hard_requirement_ids & unresolved:
        _append(
            errors,
            "requirements",
            reason="HARD_REQUIREMENTS_UNRESOLVED",
            requirement_ids=sorted(
                map(str, hard_requirement_ids & unresolved)
            ),
        )
    blueprint_process_types = {
        process.process_type for process in blueprint.processes
    }
    for process_options in required_process_type_options(profile):
        if not process_options & blueprint_process_types:
            _append(
                errors,
                "requirements",
                reason="HARD_PROCESS_REQUIREMENT_NOT_IMPLEMENTED",
                process_type_options=sorted(process_options),
            )
    required_routes = {
        str(item.expected_value).casefold()
        for item in profile.requirements
        if item.necessity == "hard_required"
        and item.category == "administration_route"
    }
    implemented_routes = {
        item.route.casefold() for item in blueprint.dosing_regimens
    }
    if not required_routes.issubset(implemented_routes):
        _append(
            errors,
            "requirements",
            reason="HARD_ROUTE_REQUIREMENT_NOT_IMPLEMENTED",
            routes=sorted(required_routes - implemented_routes),
        )
    required_objectives = {
        str(item.expected_value).casefold()
        for item in profile.requirements
        if item.necessity == "hard_required"
        and item.category == "validation_context"
    }
    if required_objectives and blueprint.objective.casefold() not in (
        required_objectives
    ):
        _append(
            errors,
            "requirements",
            reason="HARD_OBJECTIVE_REQUIREMENT_NOT_IMPLEMENTED",
        )

    compartment_ids = {
        item.compartment_id for item in blueprint.compartments
    }
    missing_required_compartments = (
        set(template.required_compartment_ids) - compartment_ids
    )
    if missing_required_compartments:
        _append(
            errors,
            "compartments",
            reason="REQUIRED_COMPARTMENTS_MISSING",
            compartment_ids=sorted(missing_required_compartments),
        )
    for compartment in blueprint.compartments:
        if (
            compartment.parent_compartment_id is not None
            and compartment.parent_compartment_id not in compartment_ids
        ):
            _append(
                errors,
                "compartments",
                reason="UNKNOWN_PARENT_COMPARTMENT",
                compartment_id=compartment.compartment_id,
                parent_compartment_id=compartment.parent_compartment_id,
            )

    equation_by_id = {
        item.equation_template_id: item
        for item in template.equation_templates
    }
    process_semantic_keys: set[tuple[object, ...]] = set()
    for process in blueprint.processes:
        if process.process_type not in template.supported_process_types:
            _append(
                errors,
                "processes",
                reason="PROCESS_TYPE_UNSUPPORTED",
                process_id=process.process_id,
                process_type=process.process_type,
            )
        equation = equation_by_id.get(process.equation_template_id)
        if equation is None:
            _append(
                errors,
                "processes",
                reason="EQUATION_TEMPLATE_UNKNOWN",
                process_id=process.process_id,
            )
        else:
            if equation.process_type != process.process_type:
                _append(
                    errors,
                    "processes",
                    reason="EQUATION_PROCESS_TYPE_MISMATCH",
                    process_id=process.process_id,
                )
            if equation.equation_dimension != process.equation_dimension:
                _append(
                    errors,
                    "dimensions",
                    reason="EQUATION_DIMENSION_MISMATCH",
                    process_id=process.process_id,
                )
            if equation.parameterization != process.parameterization:
                _append(
                    errors,
                    "processes",
                    reason="PROCESS_PARAMETERIZATION_MISMATCH",
                    process_id=process.process_id,
                )
            if not equation.mass_balance_validated:
                _append(
                    errors,
                    "mass_balance",
                    reason="EQUATION_NOT_MASS_BALANCE_VALIDATED",
                    process_id=process.process_id,
                )
        participant_roles = {
            participant.role for participant in process.participants
        }
        if "input" not in participant_roles:
            _append(
                errors,
                "mass_balance",
                reason="PROCESS_INPUT_PARTICIPANT_MISSING",
                process_id=process.process_id,
            )
        if "output" not in participant_roles:
            _append(
                errors,
                "mass_balance",
                reason="PROCESS_OUTPUT_PARTICIPANT_MISSING",
                process_id=process.process_id,
            )
        if (
            process.input_compartment_id == process.output_compartment_id
            and process.process_type
            in {
                "absorption",
                "distribution",
                "transport",
                "renal_excretion",
                "biliary_excretion",
            }
        ):
            _append(
                errors,
                "compartments",
                reason="PROCESS_SELF_CONNECTION",
                process_id=process.process_id,
            )
        if (
            isinstance(process, TransportProcess)
            and process.transport_mechanism == "carrier_mediated"
            and not process.transporter_semantic_id
        ):
            _append(
                errors,
                "processes",
                reason="CARRIER_TRANSPORTER_MISSING",
                process_id=process.process_id,
            )
        semantic_key: tuple[object, ...] = (
            process.process_type,
            process.input_compartment_id,
            process.output_compartment_id,
            process.equation_template_id,
            tuple(
                sorted(
                    (
                        item.compound_semantic_id,
                        item.role,
                        item.stoichiometric_coefficient,
                    )
                    for item in process.participants
                )
            ),
        )
        if semantic_key in process_semantic_keys:
            _append(
                errors,
                "processes",
                reason="DUPLICATE_PROCESS_MECHANISM",
                process_id=process.process_id,
            )
        process_semantic_keys.add(semantic_key)
        if isinstance(process, MetabolismProcess):
            inputs = {
                item.compound_semantic_id
                for item in process.participants
                if item.role == "input"
            }
            outputs = {
                item.compound_semantic_id
                for item in process.participants
                if item.role == "output"
            }
            if process.parent_compound_semantic_id not in inputs:
                _append(
                    errors,
                    "processes",
                    reason="METABOLISM_PARENT_DIRECTION_INVALID",
                    process_id=process.process_id,
                )
            if (
                process.metabolite_semantic_id is not None
                and process.metabolite_semantic_id not in outputs
            ):
                _append(
                    errors,
                    "processes",
                    reason="METABOLITE_OUTPUT_DIRECTION_INVALID",
                    process_id=process.process_id,
                )
        if isinstance(process, DDIProcess):
            modifiers = {
                item.compound_semantic_id
                for item in process.participants
                if item.role == "modifier"
            }
            inputs = {
                item.compound_semantic_id
                for item in process.participants
                if item.role == "input"
            }
            if process.perpetrator_compound_semantic_id not in modifiers:
                _append(
                    errors,
                    "processes",
                    reason="DDI_PERPETRATOR_DIRECTION_INVALID",
                    process_id=process.process_id,
                )
            if process.victim_compound_semantic_id not in inputs:
                _append(
                    errors,
                    "processes",
                    reason="DDI_VICTIM_DIRECTION_INVALID",
                    process_id=process.process_id,
                )

    slot_by_path = {
        item.model_path: item for item in template.parameter_slots
    }
    parameter_by_id = {
        item.parameter_id: item for item in blueprint.parameters
    }
    for parameter in blueprint.parameters:
        slot = slot_by_path.get(parameter.model_path)
        if slot is None:
            _append(
                errors,
                "parameters",
                reason="PARAMETER_PATH_NOT_WHITELISTED",
                parameter_id=parameter.parameter_id,
                model_path=parameter.model_path,
            )
        else:
            if slot.dimension != parameter.dimension:
                _append(
                    errors,
                    "dimensions",
                    reason="PARAMETER_DIMENSION_MISMATCH",
                    parameter_id=parameter.parameter_id,
                )
            if (
                parameter.canonical_unit is not None
                and slot.canonical_unit != parameter.canonical_unit
            ):
                _append(
                    errors,
                    "units",
                    reason="PARAMETER_UNIT_MISMATCH",
                    parameter_id=parameter.parameter_id,
                    blueprint_unit=parameter.canonical_unit,
                    template_unit=slot.canonical_unit,
                )
        value = getattr(parameter, "value", None)
        if value is not None:
            if not math.isfinite(value):
                _append(
                    errors,
                    "parameters",
                    reason="PARAMETER_VALUE_NON_FINITE",
                    parameter_id=parameter.parameter_id,
                )
            nonnegative_dimensions = {
                "amount",
                "mass",
                "concentration",
                "volume",
                "time",
                "flow",
                "clearance",
                "rate",
                "fraction",
                "molecular_weight",
                "permeability",
            }
            if value < 0 and parameter.dimension in nonnegative_dimensions:
                _append(
                    errors,
                    "parameters",
                    reason="PARAMETER_VALUE_NEGATIVE",
                    parameter_id=parameter.parameter_id,
                )
            if parameter.dimension == "fraction" and value > 1:
                _append(
                    errors,
                    "parameters",
                    reason="FRACTION_PARAMETER_ABOVE_ONE",
                    parameter_id=parameter.parameter_id,
                )
            if (
                slot is not None
                and slot.minimum_value is not None
                and value < slot.minimum_value
            ):
                _append(
                    errors,
                    "parameters",
                    reason="PARAMETER_BELOW_TEMPLATE_BOUND",
                    parameter_id=parameter.parameter_id,
                )
            if (
                slot is not None
                and slot.maximum_value is not None
                and value > slot.maximum_value
            ):
                _append(
                    errors,
                    "parameters",
                    reason="PARAMETER_ABOVE_TEMPLATE_BOUND",
                    parameter_id=parameter.parameter_id,
                )
        if not parameter.provenance:
            _append(
                errors,
                "provenance",
                reason="PARAMETER_PROVENANCE_MISSING",
                parameter_id=parameter.parameter_id,
            )
        if isinstance(parameter, MissingParameter) and parameter.blocking:
            _append(
                errors,
                "parameters",
                reason="BLOCKING_PARAMETER_MISSING",
                parameter_id=parameter.parameter_id,
            )
        # Estimated parameters compile with their preregistered initial value.
        # A fitted value is required only before advancing to calibrated.

    for process in blueprint.processes:
        equation = equation_by_id.get(process.equation_template_id)
        if equation is None:
            continue
        process_parameter_semantics = {
            parameter_by_id[parameter_id].semantic_id
            for parameter_id in process.parameter_ids
            if parameter_id in parameter_by_id
        }
        missing_semantics = (
            set(equation.required_parameter_semantic_ids)
            - process_parameter_semantics
        )
        if missing_semantics:
            _append(
                errors,
                "parameters",
                reason="PROCESS_REQUIRED_PARAMETERS_MISSING",
                process_id=process.process_id,
                semantic_ids=sorted(missing_semantics),
            )

    for regimen in blueprint.dosing_regimens:
        if regimen.route not in template.supported_routes:
            _append(
                errors,
                "dosing",
                reason="DOSING_ROUTE_UNSUPPORTED",
                regimen_id=regimen.regimen_id,
                route=regimen.route,
            )
    regimen_ids = {
        item.regimen_id for item in blueprint.dosing_regimens
    }
    for scenario in blueprint.simulation_scenarios:
        if scenario.regimen_id not in regimen_ids:
            _append(
                errors,
                "dosing",
                reason="SCENARIO_REGIMEN_UNKNOWN",
                scenario_id=scenario.scenario_id,
            )

    calibration_datasets = {
        item.dataset_id
        for item in blueprint.calibration_plan.datasets
    }
    validation_datasets = {
        item.dataset_id for item in blueprint.validation_plan.datasets
    }
    if calibration_datasets & validation_datasets:
        _append(
            errors,
            "datasets",
            reason="CALIBRATION_VALIDATION_DATASET_OVERLAP",
            dataset_ids=sorted(
                map(str, calibration_datasets & validation_datasets)
            ),
        )
    calibration_subjects = {
        subject
        for dataset in blueprint.calibration_plan.datasets
        for subject in dataset.subject_or_cohort_ids
    }
    validation_subjects = {
        subject
        for dataset in blueprint.validation_plan.datasets
        for subject in dataset.subject_or_cohort_ids
    }
    if calibration_subjects & validation_subjects:
        _append(
            errors,
            "datasets",
            reason="CALIBRATION_VALIDATION_SUBJECT_OVERLAP",
            subject_ids=sorted(
                calibration_subjects & validation_subjects
            ),
        )
    calibration_parameter_ids = set(
        blueprint.calibration_plan.estimated_parameter_ids
    )
    known_estimated_ids = {
        item.parameter_id
        for item in blueprint.parameters
        if isinstance(item, EstimatedParameter)
    }
    if not calibration_parameter_ids.issubset(known_estimated_ids):
        _append(
            errors,
            "datasets",
            reason="CALIBRATION_PLAN_PARAMETER_UNKNOWN",
        )

    checks = {
        category: not values
        for category, values in errors.items()
    }
    accepted = all(checks.values())
    compile_eligible = accepted and blueprint.status in {
        "draft_grounded",
        "structurally_validated",
        "compiled",
        "executable",
    }
    return BlueprintValidationReport(
        validator_version=VALIDATOR_VERSION,
        validated_at=datetime.now(UTC),
        blueprint_id=blueprint.blueprint_id,
        blueprint_checksum=blueprint.checksum,
        template_id=template.template_id,
        accepted=accepted,
        compile_eligible=compile_eligible,
        checks=checks,
        counts={
            "compartments": len(blueprint.compartments),
            "processes": len(blueprint.processes),
            "parameters": len(blueprint.parameters),
            "hard_requirements": len(hard_requirement_ids),
            "satisfied_hard_requirements": len(
                hard_requirement_ids & satisfied
            ),
            "blocking_missing_parameters": sum(
                isinstance(item, MissingParameter) and item.blocking
                for item in blueprint.parameters
            ),
            "calibration_datasets": len(calibration_datasets),
            "validation_datasets": len(validation_datasets),
            "errors": sum(len(items) for items in errors.values()),
        },
        errors=errors,
    )
