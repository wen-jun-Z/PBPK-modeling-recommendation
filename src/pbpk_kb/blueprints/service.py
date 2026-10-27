from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.api.evidence_routes import serialize_row
from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.recommendation.model_ranker import RANKER_VERSION
from pbpk_kb.recommendation.public_model_service import recommend_public_model
from pbpk_kb.recommendation.schemas import (
    RankedModelRecommendation,
    RecommendationRequest,
    RequirementProfile,
)


BLUEPRINT_SERVICE_VERSION = "registered-blueprint-service-1.4.1"


class BlueprintServiceError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _scientific_interpretation(row: dict[str, Any]) -> str:
    blockers = sorted({str(code) for code in row.get("blocker_codes") or []})
    if bool(row.get("registration_allowed")):
        return (
            "The registered scientific gates passed for this exact blueprint "
            "scope. This decision does not authorize transfer to an unregistered "
            "compound, regimen, population, formulation, endpoint, or model version."
        )
    blocker_text = ", ".join(blockers) if blockers else "UNSPECIFIED_SCIENTIFIC_BLOCKER"
    return (
        "Engineering execution succeeded, but scientific use remains blocked by "
        f"the registered scientific gates: {blocker_text}. The artifact must not "
        "be treated as calibrated, independently validated, or suitable for a "
        "dosing decision."
    )


def _normalize(value: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").casefold()))


def _normalized_values(values: list[str]) -> set[str]:
    return {_normalize(value) for value in values if _normalize(value)}


def _objective_processes(objective: str) -> set[str]:
    from pbpk_kb.recommendation.requirement_profile import (
        OBJECTIVE_PROCESS_RULES,
    )

    return {
        "renal_excretion" if value == "excretion" else value
        for value in OBJECTIVE_PROCESS_RULES[objective]
    }


def _blueprint_endpoint_set(blueprint: dict[str, Any]) -> set[str]:
    plan = blueprint.get("validation_plan") or {}
    endpoints = _normalized_values(plan.get("endpoints") or [])
    scenarios = blueprint.get("simulation_scenarios") or []
    output_paths = " ".join(
        str(path).casefold()
        for scenario in scenarios
        for path in scenario.get("output_paths") or []
    )
    if "concentration" in output_paths or "plasma" in output_paths:
        endpoints.add("concentration time profile")
    return endpoints


def _blueprint_package(root: Path) -> Path | None:
    packages = sorted(root.glob("*_blueprint_bundle.zip"))
    if len(packages) != 1:
        return None
    package = packages[0].resolve()
    try:
        package.relative_to(root.resolve())
    except ValueError:
        return None
    return package if package.is_file() else None


def _persist_ranking(
    session: Session,
    *,
    request: RecommendationRequest,
    result: dict[str, Any],
) -> None:
    profile = RequirementProfile.model_validate(result["requirement_profile"])
    ranking_payload = result.get("ranking")
    if ranking_payload is None:
        return
    ranking = RankedModelRecommendation.model_validate(ranking_payload)
    internal_model_ids = {
        str(row["external_model_version_id"]): str(row["model_version_id"])
        for row in session.execute(
            text(
                "SELECT external_model_version_id, model_version_id "
                "FROM public_pbpk_model_version"
            )
        ).mappings()
    }
    selected_internal_id = None
    if ranking.selected_model_version_id is not None:
        selected_candidate = next(
            (
                candidate
                for candidate in ranking.candidates
                if candidate.model_version_id
                == ranking.selected_model_version_id
            ),
            None,
        )
        if selected_candidate is None:
            raise BlueprintServiceError(
                "Selected ranking candidate is missing from the result."
            )
        selected_internal_id = internal_model_ids.get(
            selected_candidate.source_model_version_key
        )
        if selected_internal_id is None:
            raise BlueprintServiceError(
                "Selected model is absent from the persisted model catalog."
            )
    session.execute(
        text(
            """
            INSERT INTO requirement_profile (
                profile_id, request_id, compound_id, profile_version,
                ontology_version, ranking_eligibility, request_sha256,
                request_json, profile_checksum, profile_json,
                source_evidence_run_id, created_at
            ) VALUES (
                :profile_id, :request_id, :compound_id, :profile_version,
                :ontology_version, :ranking_eligibility, :request_sha256,
                CAST(:request_json AS jsonb), :profile_checksum,
                CAST(:profile_json AS jsonb), :source_evidence_run_id,
                :created_at
            )
            ON CONFLICT (profile_id) DO UPDATE SET
                request_sha256 = EXCLUDED.request_sha256,
                request_json = EXCLUDED.request_json,
                profile_json = EXCLUDED.profile_json,
                ranking_eligibility = EXCLUDED.ranking_eligibility
            """
        ),
        {
            "profile_id": str(profile.profile_id),
            "request_id": str(profile.request_id),
            "compound_id": str(profile.compound_identity.compound_id),
            "profile_version": profile.profile_version,
            "ontology_version": profile.ontology_version,
            "ranking_eligibility": profile.ranking_eligibility,
            "request_sha256": _sha256(request.model_dump(mode="json")),
            "request_json": _canonical_json(request.model_dump(mode="json")),
            "profile_checksum": profile.checksum,
            "profile_json": _canonical_json(profile.model_dump(mode="json")),
            "source_evidence_run_id": str(profile.source_evidence_run_id),
            "created_at": profile.created_at,
        },
    )
    session.execute(
        text(
            """
            INSERT INTO model_ranking_run (
                ranking_run_id, profile_id, decision,
                selected_model_version_id, decision_certainty,
                confidence_semantics,
                ranking_schema_version, ranker_version,
                model_run_manifest_id, result_checksum, result_json,
                created_at
            ) VALUES (
                :ranking_run_id, :profile_id, :decision,
                :selected_model_version_id, :decision_certainty,
                :confidence_semantics,
                :ranking_schema_version, :ranker_version,
                :model_run_manifest_id, :result_checksum,
                CAST(:result_json AS jsonb), :created_at
            )
            ON CONFLICT (ranking_run_id) DO UPDATE SET
                profile_id = EXCLUDED.profile_id,
                decision = EXCLUDED.decision,
                selected_model_version_id = EXCLUDED.selected_model_version_id,
                decision_certainty = EXCLUDED.decision_certainty,
                confidence_semantics = EXCLUDED.confidence_semantics,
                ranking_schema_version = EXCLUDED.ranking_schema_version,
                ranker_version = EXCLUDED.ranker_version,
                model_run_manifest_id = EXCLUDED.model_run_manifest_id,
                result_json = EXCLUDED.result_json,
                result_checksum = EXCLUDED.result_checksum,
                created_at = EXCLUDED.created_at
            """
        ),
        {
            "ranking_run_id": str(ranking.ranking_run_id),
            "profile_id": str(ranking.profile_id),
            "decision": ranking.decision,
            "selected_model_version_id": selected_internal_id,
            "decision_certainty": ranking.decision_certainty,
            "confidence_semantics": ranking.confidence_semantics,
            "ranking_schema_version": ranking.ranking_schema_version,
            "ranker_version": RANKER_VERSION,
            "model_run_manifest_id": str(ranking.model_run_manifest_id),
            "result_checksum": ranking.checksum,
            "result_json": _canonical_json(ranking.model_dump(mode="json")),
            "created_at": ranking.created_at,
        },
    )
    for rank, candidate in enumerate(ranking.candidates, start=1):
        internal_model_id = internal_model_ids.get(
            candidate.source_model_version_key
        )
        if internal_model_id is None:
            continue
        session.execute(
            text(
                """
                INSERT INTO model_ranking_candidate (
                    ranking_run_id, model_version_id, rank,
                    candidate_class, eligibility,
                    deterministic_compatibility, exclusion_reason_codes,
                    constraint_matrix_json
                ) VALUES (
                    :ranking_run_id, :model_version_id, :rank,
                    :candidate_class, :eligibility,
                    :deterministic_compatibility, :exclusion_reason_codes,
                    CAST(:constraint_matrix_json AS jsonb)
                )
                ON CONFLICT (ranking_run_id, model_version_id) DO UPDATE SET
                    rank = EXCLUDED.rank,
                    eligibility = EXCLUDED.eligibility,
                    deterministic_compatibility =
                        EXCLUDED.deterministic_compatibility,
                    exclusion_reason_codes = EXCLUDED.exclusion_reason_codes,
                    constraint_matrix_json = EXCLUDED.constraint_matrix_json
                """
            ),
            {
                "ranking_run_id": str(ranking.ranking_run_id),
                "model_version_id": internal_model_id,
                "rank": rank,
                "candidate_class": candidate.candidate_class,
                "eligibility": candidate.eligibility,
                "deterministic_compatibility": (
                    candidate.deterministic_compatibility
                ),
                "exclusion_reason_codes": candidate.exclusion_reason_codes,
                "constraint_matrix_json": _canonical_json(
                    candidate.model_dump(mode="json")
                ),
            },
        )


def _current_blueprint_row(
    session: Session,
    *,
    blueprint_id: str,
) -> dict[str, Any] | None:
    row = session.execute(
        text(
            """
            SELECT
                blueprint.blueprint_id,
                blueprint.compound_id,
                compound.preferred_name AS compound_name,
                blueprint.objective,
                blueprint.current_status,
                blueprint.scientific_status,
                blueprint.registration_allowed,
                blueprint.blocker_codes,
                blueprint.created_at,
                blueprint.updated_at,
                version.blueprint_version_id,
                version.blueprint_version,
                version.blueprint_checksum,
                version.blueprint_json,
                version.validation_report_json,
                version.compilation_report_json,
                template.template_id,
                template.external_template_id,
                template.template_version,
                template.target_runtime_adapter,
                template.runtime_version,
                template.structural_status,
                template.scientific_status AS template_scientific_status,
                template.template_eligible,
                template.scientific_registration_allowed,
                template.template_checksum,
                template.template_json,
                template.asset_root,
                template.source_location AS template_source_location,
                profile.profile_id,
                profile.profile_checksum,
                profile.request_json,
                profile.profile_json
            FROM pbpk_blueprint blueprint
            JOIN compound_identity compound
              ON compound.compound_id = blueprint.compound_id
            JOIN pbpk_blueprint_template template
              ON template.template_id = blueprint.template_id
            JOIN requirement_profile profile
              ON profile.profile_id = blueprint.profile_id
            JOIN blueprint_version version
              ON version.blueprint_id = blueprint.blueprint_id
             AND version.status = blueprint.current_status
            WHERE blueprint.blueprint_id = CAST(:blueprint_id AS uuid)
            """
        ),
        {"blueprint_id": blueprint_id},
    ).mappings().first()
    return serialize_row(dict(row)) if row else None


def _compatibility_blockers(
    *,
    request: RecommendationRequest,
    row: dict[str, Any],
) -> list[str]:
    template = row["template_json"]
    blueprint = row["blueprint_json"]
    blockers: list[str] = []
    required_contract = {
        "BLUEPRINT_REQUEST_ROUTE_MISSING": request.route,
        "BLUEPRINT_REQUEST_DOSE_MISSING": request.dose,
        "BLUEPRINT_REQUEST_DOSE_UNIT_MISSING": request.dose_unit,
        "BLUEPRINT_REQUEST_FREQUENCY_MISSING": request.frequency,
        "BLUEPRINT_REQUEST_REGIMEN_MODE_MISSING": request.single_or_multiple,
        "BLUEPRINT_REQUEST_FORMULATION_MISSING": request.formulation,
        "BLUEPRINT_REQUEST_ENDPOINT_MISSING": request.prediction_endpoints,
    }
    blockers.extend(
        code for code, value in required_contract.items() if not value
    )
    supported_objectives = _normalized_values(
        template.get("supported_objectives") or []
    )
    if _normalize(request.objective) not in supported_objectives:
        blockers.append("TEMPLATE_OBJECTIVE_UNSUPPORTED")
    if _normalize(request.species) != _normalize(blueprint["species"]):
        blockers.append("TEMPLATE_SPECIES_UNSUPPORTED")
    if request.route and _normalize(request.route) not in {
        _normalize(value) for value in template["supported_routes"]
    }:
        blockers.append("TEMPLATE_ROUTE_UNSUPPORTED")
    if request.dose is not None:
        regimen = blueprint["dosing_regimens"][0]
        if request.dose != regimen["dose"] or _normalize(
            request.dose_unit
        ) != _normalize(regimen["dose_unit"]):
            blockers.append("REQUEST_REQUIRES_TEMPLATE_RECOMPILATION")
    regimen = blueprint["dosing_regimens"][0]
    if request.frequency and _normalize(request.frequency) != _normalize(
        regimen.get("frequency")
    ):
        blockers.append("TEMPLATE_FREQUENCY_UNSUPPORTED")
    if request.single_or_multiple:
        registered_frequency = _normalize(regimen.get("frequency"))
        registered_mode = (
            "single" if "single" in registered_frequency else "multiple"
        )
        if request.single_or_multiple != registered_mode:
            blockers.append("TEMPLATE_REGIMEN_MODE_UNSUPPORTED")
    if request.infusion_duration:
        blockers.append("TEMPLATE_INFUSION_UNSUPPORTED")
    if request.fed_or_fasted not in {None, "not_applicable"}:
        blockers.append("TEMPLATE_FED_STATE_UNSUPPORTED")
    if request.population and _normalize(request.population) not in {
        "healthy adult",
        "healthy adults",
    }:
        blockers.append("TEMPLATE_POPULATION_UNSUPPORTED")
    if request.formulation:
        regimen_formulation = _normalize(
            blueprint["dosing_regimens"][0].get("formulation")
        )
        accepted_formulations = {
            regimen_formulation,
            "aqueous solution",
            "dissolved solution",
        }
        if _normalize(request.formulation) not in accepted_formulations:
            blockers.append("TEMPLATE_FORMULATION_UNSUPPORTED")
    if request.dosage_form or request.release_type:
        blockers.append("TEMPLATE_DOSAGE_FORM_UNSUPPORTED")
    process_json = [
        component["payload_json"]
        for component in row.get("process_components", [])
    ]
    enzymes = {
        _normalize(process.get("enzyme_semantic_id", "").split(":")[-1])
        for process in process_json
        if process.get("enzyme_semantic_id")
    }
    required_enzymes = {
        _normalize(value)
        for value in set(request.required_enzymes) | set(request.enzyme)
    }
    if not required_enzymes.issubset(enzymes):
        blockers.append("TEMPLATE_ENZYME_REQUIREMENT_UNSUPPORTED")
    transporters = {
        _normalize(
            process.get("transporter_semantic_id", "").split(":")[-1]
        )
        for process in process_json
        if process.get("transporter_semantic_id")
    }
    required_transporters = {
        _normalize(value)
        for value in set(request.required_transporters)
        | set(request.transporter)
    }
    if not required_transporters.issubset(transporters):
        blockers.append("TEMPLATE_TRANSPORTER_REQUIREMENT_UNSUPPORTED")
    registered_processes = {
        _normalize(process.get("process_type")) for process in process_json
    }
    for required_process in _objective_processes(request.objective):
        if required_process == "renal_excretion":
            supported = bool(
                {"renal excretion", "biliary excretion"}
                & registered_processes
            )
        else:
            supported = _normalize(required_process) in registered_processes
        if not supported:
            blockers.append(
                "TEMPLATE_REQUIRED_PROCESS_UNSUPPORTED:"
                + required_process.upper()
            )
    endpoints = _blueprint_endpoint_set(blueprint)
    for endpoint in request.prediction_endpoints:
        if _normalize(endpoint) not in endpoints:
            blockers.append("TEMPLATE_ENDPOINT_UNSUPPORTED:" + endpoint.upper())
    if request.required_organs:
        compartment_text = {
            _normalize(value.get("name") or value.get("semantic_id"))
            for value in blueprint.get("compartments") or []
        }
        for organ in request.required_organs:
            normalized = _normalize(organ)
            if not any(
                normalized in compartment or compartment in normalized
                for compartment in compartment_text
            ):
                blockers.append("TEMPLATE_ORGAN_UNSUPPORTED:" + organ.upper())
    requested_structure = request.whole_body_or_reduced_pbpk
    if requested_structure and requested_structure != "unspecified":
        if _normalize(requested_structure) != _normalize(
            template.get("model_structure")
        ):
            blockers.append("TEMPLATE_MODEL_STRUCTURE_UNSUPPORTED")
    unsupported_explicit_fields = {
        "TEMPLATE_SALT_FORM_UNSUPPORTED": request.salt_form,
        "TEMPLATE_METABOLITE_SCOPE_UNSUPPORTED": request.metabolites_to_model,
        "TEMPLATE_CANDIDATE_MODEL_SELECTOR_UNSUPPORTED": (
            request.candidate_public_model
        ),
        "TEMPLATE_SEX_SCOPE_UNSUPPORTED": (
            request.sex if request.sex not in {None, "unspecified"} else None
        ),
        "TEMPLATE_BODY_WEIGHT_SCOPE_UNSUPPORTED": request.body_weight_kg,
        "TEMPLATE_ETHNICITY_SCOPE_UNSUPPORTED": request.ethnicity,
        "TEMPLATE_SPECIAL_POPULATION_SCOPE_UNSUPPORTED": any(
            (
                request.age_range,
                request.renal_function,
                request.hepatic_function,
                request.disease_state,
                request.genotype,
                request.pregnancy_status,
            )
        ),
        "TEMPLATE_PARENT_METABOLITE_SCOPE_UNSUPPORTED": (
            request.parent_or_parent_metabolite
        ),
        "TEMPLATE_SOFTWARE_PREFERENCE_UNSUPPORTED": bool(
            request.software_preference
            and not all(
                "pk sim" in _normalize(value)
                for value in request.software_preference
            )
        ),
        "TEMPLATE_DDI_SCOPE_UNSUPPORTED": any(
            (
                request.victim_drug,
                request.perpetrator_drug,
                request.interacting_drug,
                request.interaction_mechanism,
                request.interaction_duration,
                request.victim_regimen,
                request.perpetrator_regimen,
            )
        ),
        "TEMPLATE_DATASET_CONTRACT_UNSUPPORTED": any(
            (
                request.calibration_datasets,
                request.validation_datasets,
                request.acceptance_criteria,
            )
        ),
        "TEMPLATE_ANALYSIS_CONTRACT_UNSUPPORTED": any(
            (
                request.parameter_uncertainty,
                request.sensitivity_analysis,
                request.uncertainty_analysis,
                request.constraints,
            )
        ),
    }
    blockers.extend(
        code for code, value in unsupported_explicit_fields.items() if value
    )
    return sorted(set(blockers))


def _select_compatible_blueprint(
    *,
    request: RecommendationRequest,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]]]:
    audits: list[dict[str, Any]] = []
    compatible: list[dict[str, Any]] = []
    all_blockers: set[str] = set()
    for row in rows:
        blockers = _compatibility_blockers(request=request, row=row)
        all_blockers.update(blockers)
        audits.append(
            {
                "blueprint_id": row.get("blueprint_id"),
                "template_id": row.get("template_id"),
                "template_checksum": row.get("template_checksum"),
                "blueprint_checksum": row.get("blueprint_checksum"),
                "scientific_status": row.get("scientific_status"),
                "registration_allowed": bool(row.get("registration_allowed")),
                "compatible": not blockers,
                "blockers": blockers,
            }
        )
        if not blockers:
            compatible.append(row)
    if len(compatible) == 1:
        return compatible[0], [], audits
    if len(compatible) > 1:
        return None, ["AMBIGUOUS_COMPATIBLE_BLUEPRINT_TEMPLATES"], audits
    return None, sorted(all_blockers), audits


def _load_process_components(
    session: Session,
    *,
    version_id: str,
) -> list[dict[str, Any]]:
    return [
        serialize_row(dict(row))
        for row in session.execute(
            text(
                """
                SELECT component_id, payload_json
                FROM blueprint_component
                WHERE blueprint_version_id = CAST(:version_id AS uuid)
                  AND component_kind = 'process'
                ORDER BY component_id
                """
            ),
            {"version_id": version_id},
        ).mappings()
    ]


def create_blueprint_for_request(
    *,
    session: Session,
    request: RecommendationRequest,
) -> dict[str, Any]:
    ranking = recommend_public_model(session=session, request=request)
    _persist_ranking(session, request=request, result=ranking)
    if ranking["decision"] == "reuse":
        session.commit()
        return {
            "status": "model_reuse_selected",
            "decision": "reuse",
            "blueprint": None,
            "ranking": ranking,
            "reason_codes": ["BLUEPRINT_NOT_REQUIRED"],
            "service_version": BLUEPRINT_SERVICE_VERSION,
        }
    compound_id = ranking["requirement_profile"]["compound_identity"][
        "compound_id"
    ]
    template_rows = session.execute(
        text(
            """
            SELECT blueprint.blueprint_id
            FROM pbpk_blueprint_template template
            JOIN pbpk_blueprint blueprint
              ON blueprint.template_id = template.template_id
            WHERE template.compound_id = CAST(:compound_id AS uuid)
              AND template.template_eligible IS TRUE
              AND blueprint.current_status = 'executable'
            ORDER BY template.updated_at DESC, blueprint.updated_at DESC
            """
        ),
        {"compound_id": compound_id},
    ).scalars().all()
    if not template_rows:
        session.commit()
        return {
            "status": "abstain_no_executable_template",
            "decision": "abstain",
            "blueprint": None,
            "ranking": ranking,
            "reason_codes": ["NO_EXACT_COMPOUND_EXECUTABLE_TEMPLATE"],
            "service_version": BLUEPRINT_SERVICE_VERSION,
        }
    candidate_rows: list[dict[str, Any]] = []
    for blueprint_id in template_rows:
        candidate_row = _current_blueprint_row(
            session,
            blueprint_id=str(blueprint_id),
        )
        if candidate_row is None:
            raise BlueprintServiceError("Registered blueprint cannot be loaded.")
        candidate_row["process_components"] = _load_process_components(
            session,
            version_id=candidate_row["blueprint_version_id"],
        )
        candidate_rows.append(candidate_row)
    row, blockers, template_audits = _select_compatible_blueprint(
        request=request,
        rows=candidate_rows,
    )
    for candidate_row in candidate_rows:
        candidate_row.pop("process_components", None)
    if row is None:
        session.commit()
        return {
            "status": "abstain_template_incompatible",
            "decision": "abstain",
            "blueprint": None,
            "ranking": ranking,
            "reason_codes": blockers,
            "compatibility_audit": {
                "policy_version": BLUEPRINT_SERVICE_VERSION,
                "accepted": False,
                "request_sha256": _sha256(request.model_dump(mode="json")),
                "evaluated_template_count": len(candidate_rows),
                "template_candidates": template_audits,
                "blockers": blockers,
            },
            "service_version": BLUEPRINT_SERVICE_VERSION,
        }
    session.commit()
    return {
        "status": "executable_template_available",
        "decision": "blueprint",
        "blueprint_mode": "registered_executable_baseline",
        "engineering_execution_only": True,
        "engineering_decision": "return_registered_executable_template",
        "scientific_use_allowed": bool(row["registration_allowed"]),
        "scientific_decision": (
            "eligible_within_registered_validation_scope"
            if row["registration_allowed"]
            else "abstain"
        ),
        "scientific_blocker_codes": list(row["blocker_codes"] or []),
        "blueprint": row,
        "ranking": ranking,
        "reason_codes": [],
        "compatibility_audit": {
            "policy_version": BLUEPRINT_SERVICE_VERSION,
            "accepted": True,
            "request_sha256": _sha256(request.model_dump(mode="json")),
            "template_checksum": row["template_checksum"],
            "blueprint_checksum": row["blueprint_checksum"],
            "evaluated_template_count": len(candidate_rows),
            "template_candidates": template_audits,
            "blockers": [],
        },
        "scientific_interpretation": _scientific_interpretation(row),
        "service_version": BLUEPRINT_SERVICE_VERSION,
    }


def list_blueprints(
    session: Session,
    *,
    compound: str = "",
) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT blueprint.blueprint_id, compound.preferred_name AS compound_name,
                   blueprint.objective, blueprint.current_status,
                   blueprint.scientific_status,
                   blueprint.registration_allowed,
                   blueprint.blocker_codes,
                   template.external_template_id, template.template_version,
                   template.target_runtime_adapter, template.runtime_version,
                   blueprint.updated_at
            FROM pbpk_blueprint blueprint
            JOIN compound_identity compound
              ON compound.compound_id = blueprint.compound_id
            JOIN pbpk_blueprint_template template
              ON template.template_id = blueprint.template_id
            WHERE (:compound = '' OR lower(compound.preferred_name::text) =
                   lower(:compound))
            ORDER BY lower(compound.preferred_name::text), blueprint.updated_at DESC
            """
        ),
        {"compound": compound.strip()},
    ).mappings()
    return [serialize_row(dict(row)) for row in rows]


def get_blueprint_detail(session: Session, blueprint_id: str) -> dict[str, Any] | None:
    row = _current_blueprint_row(session, blueprint_id=blueprint_id)
    if row is None:
        return None
    counts = session.execute(
        text(
            """
            SELECT
              (SELECT COUNT(*) FROM blueprint_component
               WHERE blueprint_version_id = CAST(:version_id AS uuid)
                 AND component_kind = 'compartment') AS compartments,
              (SELECT COUNT(*) FROM blueprint_component
               WHERE blueprint_version_id = CAST(:version_id AS uuid)
                 AND component_kind = 'process') AS processes,
              (SELECT COUNT(*) FROM blueprint_parameter
               WHERE blueprint_version_id = CAST(:version_id AS uuid)) AS parameters,
              (SELECT COUNT(*) FROM blueprint_artifact
               WHERE blueprint_version_id = CAST(:version_id AS uuid)) AS artifacts,
              (SELECT COUNT(*) FROM simulation_run
               WHERE blueprint_version_id = CAST(:version_id AS uuid)) AS simulation_runs,
              (SELECT COUNT(*) FROM simulation_output output
               JOIN simulation_run run
                 ON run.simulation_run_id = output.simulation_run_id
               WHERE run.blueprint_version_id = CAST(:version_id AS uuid)) AS outputs
            """
        ),
        {"version_id": row["blueprint_version_id"]},
    ).mappings().one()
    package = _blueprint_package(Path(row["asset_root"]))
    row["counts"] = {key: int(value) for key, value in counts.items()}
    row["package"] = {
        "relative_path": package.name if package else None,
        "available": package is not None,
        "sha256": sha256_file(package) if package else None,
        "byte_size": package.stat().st_size if package else None,
    }
    return row
