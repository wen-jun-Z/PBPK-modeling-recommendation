from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import text
from sqlalchemy.orm import Session

from pbpk_kb.paths import PROJECT_ROOT, model_matching_root, resolve_recorded_path
from pbpk_kb.recommendation.model_candidate_retrieval import (
    CANDIDATE_LAYER_ORDER,
    candidate_layer_decision,
    retrieve_model_candidates,
)
from pbpk_kb.recommendation.model_matching import (
    build_model_graph,
    build_requirement_graph,
)
from pbpk_kb.recommendation.model_matching.serving import (
    score_for_serving,
    serving_status,
)
from pbpk_kb.recommendation.model_catalog import (
    MODEL_CATALOG_LOADER_VERSION,
    load_registered_model_snapshots,
)
from pbpk_kb.recommendation.model_ranker import (
    RANKER_VERSION,
    build_ranking_recommendation,
)
from pbpk_kb.recommendation.model_relevance import (
    MODEL_RELEVANCE_POLICY_VERSION,
    candidate_relevance_decision,
    model_mechanism_index_from_environment,
    target_mechanisms_for_request,
    unavailable_model_mechanism_index,
)
from pbpk_kb.recommendation.requirement_profile import (
    build_requirement_profile,
)
from pbpk_kb.recommendation.schemas import (
    CompoundRef,
    RecommendationRequest,
)


PUBLIC_MODEL_SERVICE_VERSION = "pbpk-public-model-service-1.7.0"
ONTOLOGY_VERSION = "pbpk-recommendation-ontology-1.1.0"


class PublicModelRecommendationError(RuntimeError):
    pass


def _stable_uuid(kind: str, *parts: object) -> UUID:
    payload = json.dumps(
        [kind, *parts],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return uuid5(NAMESPACE_URL, f"pbpk-public-model-service:{payload}")


def _resolve_compound(
    session: Session,
    compound_name: str,
) -> dict[str, Any]:
    rows = session.execute(
        text(
            """
            SELECT
                compound_id::text,
                preferred_name::text,
                normalized_name::text,
                pubchem_cid,
                chembl_id,
                drugcentral_id,
                rxnorm_rxcui,
                inchi_key
            FROM compound_identity
            WHERE lower(preferred_name::text) = lower(:compound_name)
               OR lower(COALESCE(normalized_name::text, '')) = lower(:compound_name)
               OR EXISTS (
                    SELECT 1 FROM unnest(synonyms) AS synonym
                    WHERE lower(synonym) = lower(:compound_name)
               )
            ORDER BY
                (lower(preferred_name::text) = lower(:compound_name)) DESC,
                compound_id
            """
        ),
        {"compound_name": compound_name.strip()},
    ).mappings().all()
    if not rows:
        raise PublicModelRecommendationError(
            f"Compound is not indexed: {compound_name}"
        )
    exact_ids = {str(row["compound_id"]) for row in rows}
    if len(exact_ids) != 1:
        raise PublicModelRecommendationError(
            f"Compound identity is ambiguous: {compound_name}"
        )
    return dict(rows[0])


def _catalog_timestamp(session: Session) -> datetime:
    value = session.execute(
        text(
            """
            SELECT MAX(validated_at)
            FROM public_pbpk_model_validation_run
            WHERE scientific_accepted IS TRUE
            """
        )
    ).scalar_one_or_none()
    return value or datetime.now(UTC)


def _catalog_model_rows(session: Session) -> dict[str, dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT
                version.external_model_version_id,
                version.model_version_id::text,
                registry.model_id::text,
                registry.model_name,
                registry.compound_id::text,
                compound.preferred_name::text AS compound_name,
                version.repository_url,
                version.repository_commit,
                version.version_label AS release_tag,
                registry.license_spdx_id,
                registry.source_id,
                version.source_location,
                registry.validation_grade,
                registry.run_label_grade AS catalog_run_label_grade,
                registry.ranking_label AS catalog_ranking_label,
                registry.reusable_candidate AS catalog_reusable_candidate,
                version.project_relative_path,
                version.project_sha256,
                version.source_snapshot_sha256,
                version.software,
                version.software_version,
                version.license_status,
                version.license_spdx_id AS version_license_spdx_id,
                version.license_source_location,
                version.static_parse_status,
                version.execution_probe_status,
                version.pkml_parse_status,
                version.compounds,
                version.species,
                version.populations,
                version.routes,
                version.formulations,
                version.enzymes,
                version.transporters,
                version.process_types,
                version.output_names,
                version.simulation_count,
                version.parameter_record_count,
                version.component_record_count,
                version.process_record_count,
                version.report_count,
                version.dataset_reference_count,
                version.metadata_json AS version_metadata
            FROM public_pbpk_model_version version
            JOIN public_pbpk_model_registry registry
              ON registry.model_id = version.model_id
            LEFT JOIN compound_identity compound
              ON compound.compound_id = registry.compound_id
            ORDER BY registry.model_name, version.repository_commit
            """
        )
    ).mappings()
    return {
        str(row["external_model_version_id"]): dict(row)
        for row in rows
    }


def public_model_catalog_summary(session: Session) -> dict[str, Any]:
    counts = session.execute(
        text(
            """
            SELECT
                COUNT(DISTINCT registry.model_id) AS model_count,
                COUNT(DISTINCT version.model_version_id) AS model_version_count,
                COUNT(DISTINCT asset.model_asset_id) AS asset_count,
                COUNT(DISTINCT simulation.model_simulation_id) AS simulation_count,
                COUNT(DISTINCT version.model_version_id) FILTER (
                    WHERE version.execution_probe_status = 'passed'
                      AND version.pkml_parse_status = 'parsed'
                ) AS runtime_verified_version_count,
                COUNT(DISTINCT version.model_version_id) FILTER (
                    WHERE version.reusable_candidate IS TRUE
                ) AS reusable_version_count
            FROM public_pbpk_model_registry registry
            JOIN public_pbpk_model_version version
              ON version.model_id = registry.model_id
            LEFT JOIN public_pbpk_model_asset asset
              ON asset.model_version_id = version.model_version_id
            LEFT JOIN public_pbpk_model_simulation simulation
              ON simulation.model_version_id = version.model_version_id
            """
        )
    ).mappings().one()
    compounds = [
        {
            "compound_id": str(row["compound_id"]),
            "compound_name": str(row["compound_name"]),
            "same_compound_model_version_count": int(
                row["same_compound_model_version_count"]
            ),
            "exact_family_model_version_count": int(
                row["exact_family_model_version_count"]
            ),
            "reusable_model_version_count": int(
                row["reusable_model_version_count"]
            ),
        }
        for row in session.execute(
            text(
                """
                SELECT
                    compound.compound_id,
                    compound.preferred_name::text AS compound_name,
                    COUNT(DISTINCT version.model_version_id) AS
                        same_compound_model_version_count,
                    COUNT(DISTINCT version.model_version_id) FILTER (
                        WHERE lower(
                            COALESCE(
                                version.metadata_json->>'model_family',
                                ''
                            )
                        ) = lower(compound.preferred_name::text)
                    ) AS exact_family_model_version_count,
                    COUNT(DISTINCT version.model_version_id) FILTER (
                        WHERE version.reusable_candidate IS TRUE
                    ) AS reusable_model_version_count
                FROM compound_identity compound
                LEFT JOIN public_pbpk_model_version version
                  ON EXISTS (
                      SELECT 1
                      FROM unnest(version.compounds) AS model_compound
                      WHERE lower(model_compound) =
                            lower(compound.preferred_name::text)
                  )
                GROUP BY compound.compound_id, compound.preferred_name
                ORDER BY lower(compound.preferred_name::text)
                """
            )
        ).mappings()
    ]
    blueprint_template_count = int(
        session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM pbpk_blueprint_template
                WHERE template_eligible IS TRUE
                """
            )
        ).scalar_one()
    )
    return {
        **{key: int(value or 0) for key, value in counts.items()},
        "compounds": compounds,
        "decision_policy": "public_model_first_then_exact_compound_blueprint",
        "blueprint_generation_enabled": blueprint_template_count > 0,
        "executable_blueprint_template_count": blueprint_template_count,
        "automatic_reuse_gate": (
            "Exactly one same-compound Grade 3 reuse-positive candidate "
            "must pass every hard constraint."
        ),
    }


def public_model_matching_evaluation() -> dict[str, Any]:
    """Expose the current expanded catalog experiment and its label limitations."""
    configured = os.getenv("PBPK_MODEL_MATCHING_EVALUATION_MANIFEST", "").strip()
    candidates = (
        [resolve_recorded_path(configured)]
        if configured
        else sorted(
            model_matching_root().glob("expanded_catalog_*/benchmark/experiment_manifest.json"),
            key=lambda path: path.parent.parent.name,
            reverse=True,
        )
    )
    manifest = next((path for path in candidates if path.is_file()), None)
    unavailable = {
        "status": "not_available",
        "benchmark_level": None,
        "scientific_claim_eligible": False,
        "reason": "No expanded public-model ranking experiment is available.",
    }
    if manifest is None:
        return unavailable
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if value.get("benchmark_level") != "expanded_metadata_ranking":
        return unavailable
    value["model_count"] = value.get("candidate_count")
    value["status"] = "available"
    value["manifest_path"] = str(manifest)
    value["serving"] = serving_status()
    run = manifest.parent.parent if manifest.parent.name == "benchmark" else manifest.parent
    status_path, results_path = run / "status.json", run / "results.json"
    value["training_status"] = json.loads(status_path.read_text()) if status_path.is_file() else None
    value["results"] = json.loads(results_path.read_text()) if results_path.is_file() else None
    value["evaluation_validity"] = "compatibility_rule_fit_not_independent_model_quality"
    value["evaluation_notice"] = (
        "Labels come from evidence and model matching rules that are also represented in the inputs. "
        "High ranking scores measure agreement with those rules and do not establish "
        "scientific model applicability."
    )
    held_out_compounds = (value.get("compound_split_counts") or {}).get("test")
    if held_out_compounds is not None:
        value["evaluation_notice"] += f" The held-out set contains {held_out_compounds} compounds."
    value["audit_available"] = (run / "lambdamart_audit.json").is_file()
    policy_path = run / "evaluation_policy.json"
    if policy_path.is_file():
        policy = json.loads(policy_path.read_text())
        value["evaluation_status"] = policy.get("status")
        if policy.get("allow_training") is False:
            value["evaluation_notice"] = (
                "This experiment is retired and training is disabled. Results are diagnostic only. "
                + value["evaluation_notice"]
            )
    return value


def recommend_public_model(
    *,
    session: Session,
    request: RecommendationRequest,
) -> dict[str, Any]:
    compound = _resolve_compound(session, request.compound)
    models = load_registered_model_snapshots(session)
    catalog_rows = _catalog_model_rows(session)
    catalog_time = _catalog_timestamp(session)
    identity = CompoundRef(
        compound_id=UUID(compound["compound_id"]),
        preferred_name=str(compound["preferred_name"]),
        active_moiety=str(compound["preferred_name"]),
        external_identifiers={
            key: str(value)
            for key, value in {
                "pubchem_cid": compound.get("pubchem_cid"),
                "chembl_id": compound.get("chembl_id"),
                "drugcentral_id": compound.get("drugcentral_id"),
                "rxnorm_rxcui": compound.get("rxnorm_rxcui"),
                "inchi_key": compound.get("inchi_key"),
            }.items()
            if value
        },
        identity_resolution_method="database_exact_id",
        identity_confidence=1.0,
    )
    source_run_id = _stable_uuid(
        "catalog-evidence-run",
        *sorted(
            model.validation_run_label.validation_run_id
            for model in models
            if model.validation_run_label is not None
        ),
    )
    profile = build_requirement_profile(
        request=request,
        compound_identity=identity,
        source_evidence_run_id=source_run_id,
        ontology_version=ONTOLOGY_VERSION,
        created_at=catalog_time,
    )
    requirement_graph = build_requirement_graph(profile)
    if not models:
        return {
            "status": "public_model_catalog_empty",
            "decision": "abstain",
            "request": request.model_dump(mode="json"),
            "requirement_profile": profile.model_dump(mode="json"),
            "candidates": [],
            "selected_model": None,
            "top_same_compound_candidate": None,
            "candidate_summary": {
                "retrieved": 0,
                "catalog_scanned": 0,
                "relevant_candidates": 0,
                "excluded_unrelated": 0,
                "same_compound": 0,
                "mechanistic_template": 0,
                "reuse_ready": 0,
                "validated_reuse": 0,
                "controlled_adaptation": 0,
                "same_compound_qualification": 0,
                "mechanistic_reference": 0,
                "blocked": 0,
            },
            "candidates_by_layer": {
                layer: [] for layer in CANDIDATE_LAYER_ORDER
            },
            "excluded_unrelated_models": [],
            "relevance_audit": {
                "policy_version": MODEL_RELEVANCE_POLICY_VERSION,
                "target_mechanisms": [],
                "model_kg": {
                    "status": "not_evaluated_catalog_empty",
                    "reason_code": "PUBLIC_MODEL_CATALOG_EMPTY",
                },
            },
            "abstention_reason_codes": ["NO_ELIGIBLE_MODEL"],
            "blueprint_status": "eligible_for_template_lookup",
            "methodology": _methodology(),
        }

    catalog_candidates = retrieve_model_candidates(
        profile=profile,
        models=models,
    )
    model_by_version_key = {
        model.model_version_id: model for model in models
    }
    target_mechanisms = target_mechanisms_for_request(
        session=session,
        compound_id=compound["compound_id"],
        request=request,
    )
    mechanism_index = (
        model_mechanism_index_from_environment(
            expected_model_version_count=len(models),
        )
        if target_mechanisms
        else unavailable_model_mechanism_index(
            "MODEL_KG_NOT_NEEDED_NO_TARGET_MECHANISM"
        )
    )
    relevance_by_version: dict[str, dict[str, Any]] = {}
    matching_by_version: dict[str, dict[str, Any]] = {}
    relevant_candidates = []
    excluded_unrelated_models: list[dict[str, Any]] = []
    for candidate in catalog_candidates:
        model = model_by_version_key[candidate.source_model_version_key]
        matching_by_version[candidate.source_model_version_key] = score_for_serving(
            requirement_graph,
            build_model_graph(model),
        )
        relevance = candidate_relevance_decision(
            profile=profile,
            model=model,
            candidate=candidate,
            target_mechanisms=target_mechanisms,
            mechanism_index=mechanism_index,
        )
        relevance_by_version[candidate.source_model_version_key] = relevance
        if relevance["is_relevant"]:
            relevant_candidates.append(candidate)
            continue
        catalog = catalog_rows.get(candidate.source_model_version_key) or {}
        excluded_unrelated_models.append(
            {
                "model_id": catalog.get("model_id"),
                "model_version_id": catalog.get("model_version_id"),
                "external_model_version_id": (
                    candidate.source_model_version_key
                ),
                "model_name": catalog.get("model_name")
                or candidate.model_family,
                "project_relative_path": catalog.get(
                    "project_relative_path"
                ),
                "supported_compounds": catalog.get("compounds") or [],
                "relevance_class": relevance["relevance_class"],
                "relevance_reason_codes": relevance[
                    "relevance_reason_codes"
                ],
            }
        )
    if len(relevant_candidates) + len(excluded_unrelated_models) != len(
        catalog_candidates
    ):
        raise PublicModelRecommendationError(
            "Model relevance partition is incomplete."
        )

    grade3_exact = [
        candidate
        for candidate in relevant_candidates
        if candidate.candidate_class == "exact_compound_model"
        and candidate.eligibility == "eligible"
        and candidate.run_label_grade == 3
        and candidate.ranking_label == "reuse_positive"
    ]
    model_run_manifest_id = _stable_uuid(
        "catalog-model-run",
        *sorted(
            f"{model.model_version_id}:"
            f"{model.validation_run_label.validation_run_id if model.validation_run_label else ''}"
            for model in models
        ),
    )
    ranking = (
        build_ranking_recommendation(
            profile=profile,
            candidates=relevant_candidates,
            model_run_manifest_id=model_run_manifest_id,
            auto_reuse_enabled=len(grade3_exact) == 1,
            created_at=catalog_time,
        )
        if relevant_candidates
        else None
    )
    decision = ranking.decision if ranking is not None else "abstain"
    selected = None
    if ranking is not None and ranking.selected_model_version_id is not None:
        selected_candidate = next(
            candidate
            for candidate in ranking.candidates
            if candidate.model_version_id == ranking.selected_model_version_id
        )
        selected = catalog_rows.get(
            selected_candidate.source_model_version_key
        )
        if selected is None:
            raise PublicModelRecommendationError(
                "Selected model version is absent from the database catalog."
            )
        selected = _selected_model_details(
            session,
            selected,
            selected_candidate.model_dump(mode="json"),
        )
        selected_snapshot = model_by_version_key[
            selected_candidate.source_model_version_key
        ]
        selected["validation_scope"] = (
            selected_snapshot.validation_scope.model_dump(mode="json")
            if selected_snapshot.validation_scope is not None
            else None
        )
    candidate_catalog: list[dict[str, Any]] = []
    for candidate in ranking.candidates if ranking is not None else []:
        catalog = catalog_rows.get(candidate.source_model_version_key) or {}
        model = model_by_version_key[candidate.source_model_version_key]
        relevance = relevance_by_version[candidate.source_model_version_key]
        layer = candidate_layer_decision(
            profile=profile,
            model=model,
            candidate=candidate,
            relevance_decision=relevance,
        )
        hard_blockers = [
            {
                "constraint_id": check.constraint_id,
                "result": check.result,
                "reason_code": check.reason_code,
                "observed_value": check.observed_value,
            }
            for check in candidate.constraint_checks
            if check.severity == "hard"
            and check.result in {"fail", "unknown"}
        ]
        validation_scope_check = next(
            (
                check
                for check in candidate.constraint_checks
                if ":validation_scope_transferability:"
                in check.constraint_id
            ),
            None,
        )
        numeric_transferability = (
            validation_scope_check.observed_value.get(
                "numeric_transferability"
            )
            if validation_scope_check is not None
            and isinstance(validation_scope_check.observed_value, dict)
            else None
        )
        asset_locations = sorted(
            {
                location
                for check in candidate.constraint_checks
                for location in check.model_asset_locations
            }
        )
        candidate_catalog.append(
            {
                "rank": None,
                "model_id": catalog.get("model_id"),
                "model_version_id": catalog.get("model_version_id"),
                "external_model_version_id": (
                    candidate.source_model_version_key
                ),
                "model_name": catalog.get("model_name")
                or candidate.model_family,
                "compound_name": catalog.get("compound_name"),
                "supported_compounds": catalog.get("compounds") or [],
                "candidate_class": candidate.candidate_class,
                "eligibility": candidate.eligibility,
                **relevance,
                "recommendation_tier": (
                    "reuse_ready"
                    if candidate.eligibility == "eligible"
                    and candidate.run_label_grade == 3
                    and candidate.ranking_label == "reuse_positive"
                    else "same_compound_reuse_blocked"
                    if candidate.candidate_class
                    in {"exact_compound_model", "same_compound_adaptation"}
                    else "mechanistic_template_only"
                ),
                **layer,
                "scenario_transferability": numeric_transferability,
                "scenario_transferability_class": (
                    numeric_transferability.get("match_class")
                    if isinstance(numeric_transferability, dict)
                    else None
                ),
                "deterministic_compatibility": (
                    candidate.deterministic_compatibility
                ),
                "cross_graph_matching": matching_by_version[
                    candidate.source_model_version_key
                ],
                "matching_score": matching_by_version[
                    candidate.source_model_version_key
                ]["score"],
                "adaptation_cost": matching_by_version[
                    candidate.source_model_version_key
                ]["adaptation_cost"],
                # Validation belongs to this version; grade zero must not
                # inherit acceptance from another version in the same family.
                "run_label_grade": candidate.run_label_grade,
                "ranking_label": candidate.ranking_label,
                "exclusion_reason_codes": candidate.exclusion_reason_codes,
                "hard_blockers": hard_blockers,
                "matched_requirement_count": len(
                    candidate.matched_requirement_ids
                ),
                "unmet_requirement_count": len(
                    candidate.unmet_requirement_ids
                ),
                "software": catalog.get("software"),
                "software_version": catalog.get("software_version"),
                "species": catalog.get("species") or [],
                "populations": catalog.get("populations") or [],
                "routes": catalog.get("routes") or [],
                "formulations": catalog.get("formulations") or [],
                "enzymes": catalog.get("enzymes") or [],
                "transporters": catalog.get("transporters") or [],
                "process_types": catalog.get("process_types") or [],
                "project_relative_path": catalog.get(
                    "project_relative_path"
                ),
                "project_sha256": catalog.get("project_sha256"),
                "source_snapshot_sha256": catalog.get(
                    "source_snapshot_sha256"
                ),
                "repository_url": catalog.get("repository_url"),
                "repository_commit": catalog.get("repository_commit"),
                "source_id": catalog.get("source_id"),
                "source_location": catalog.get("source_location"),
                "license_status": catalog.get("license_status"),
                "license_spdx_id": catalog.get("version_license_spdx_id"),
                "static_parse_status": catalog.get(
                    "static_parse_status"
                ),
                "pkml_parse_status": catalog.get("pkml_parse_status"),
                "execution_probe_status": catalog.get(
                    "execution_probe_status"
                ),
                "simulation_count": catalog.get("simulation_count") or 0,
                "dataset_reference_count": catalog.get(
                    "dataset_reference_count"
                ) or 0,
                "asset_locations": asset_locations,
                "validation_scope": (
                    model.validation_scope.model_dump(mode="json")
                    if model.validation_scope is not None
                    else None
                ),
            }
        )
    candidate_catalog.sort(
        key=lambda candidate: (
            int(candidate["candidate_layer_order"]),
            -float(candidate.get("matching_score") or 0.0),
            -float(candidate.get("deterministic_compatibility") or 0.0),
            str(candidate.get("model_name") or "").casefold(),
            str(candidate.get("external_model_version_id") or ""),
        )
    )
    for rank, candidate in enumerate(candidate_catalog, 1):
        candidate["rank"] = rank

    candidates_by_layer = {
        layer: [
            candidate
            for candidate in candidate_catalog
            if candidate["candidate_layer"] == layer
        ]
        for layer in CANDIDATE_LAYER_ORDER
    }
    same_compound_candidates = [
        candidate
        for candidate in candidate_catalog
        if candidate["candidate_class"]
        in {"exact_compound_model", "same_compound_adaptation"}
    ]
    mechanistic_templates = [
        candidate
        for candidate in candidate_catalog
        if candidate["candidate_class"] == "mechanistic_template"
    ]
    reuse_ready = [
        candidate
        for candidate in same_compound_candidates
        if candidate["recommendation_tier"] == "reuse_ready"
    ]
    recommended_model_candidate = next(
        (
            candidate
            for candidate in same_compound_candidates
            if candidate["candidate_layer"] != "blocked"
        ),
        None,
    )
    recommended_use = (
        "direct_reuse"
        if recommended_model_candidate
        and recommended_model_candidate["candidate_layer"]
        == "validated_reuse"
        else "controlled_adaptation"
        if recommended_model_candidate
        and recommended_model_candidate["candidate_layer"]
        == "controlled_adaptation_candidate"
        else "qualification_required"
        if recommended_model_candidate is not None
        else "no_same_compound_recommendation"
    )
    reason_codes = (
        []
        if decision == "reuse"
        else sorted(
            {
                "NO_ELIGIBLE_MODEL",
                *(
                    reason
                    for candidate in same_compound_candidates
                    for reason in candidate["exclusion_reason_codes"]
                ),
            }
        )
        if same_compound_candidates
        else ["NO_EXACT_COMPOUND_MODEL"]
        if mechanistic_templates
        else ["NO_RELEVANT_PUBLIC_MODEL"]
    )
    status = (
        "model_reuse_recommended"
        if decision == "reuse"
        else "model_adaptation_recommended"
        if recommended_use == "controlled_adaptation"
        else "model_qualification_recommended"
        if recommended_use == "qualification_required"
        else "same_compound_candidates_blocked"
        if same_compound_candidates
        else "mechanistic_references_found_no_same_compound_model"
        if mechanistic_templates
        else "no_relevant_public_model"
    )
    relevance_audit = {
        "policy_version": MODEL_RELEVANCE_POLICY_VERSION,
        "target_mechanisms": [
            {
                "mechanism_type": mechanism.mechanism_type,
                "canonical_name": mechanism.canonical_name,
                "aliases": list(mechanism.aliases),
                "origins": list(mechanism.origins),
                "relationship_types": list(
                    mechanism.relationship_types
                ),
                "relationship_ids": list(mechanism.relationship_ids),
                "evidence_ids": list(mechanism.evidence_ids),
                "source_locations": list(mechanism.source_locations),
            }
            for mechanism in target_mechanisms
        ],
        "model_kg": {
            "status": mechanism_index.status,
            "reason_code": mechanism_index.reason_code,
            "graph_summary_sha256": (
                mechanism_index.graph_summary_sha256
            ),
            "graph_sqlite_sha256": (
                mechanism_index.graph_sqlite_sha256
            ),
            "model_version_count": mechanism_index.model_version_count,
            "verified_process_path_count": len(mechanism_index.paths),
        },
    }
    return {
        "status": status,
        "decision": decision,
        "request": request.model_dump(mode="json"),
        "requirement_profile": profile.model_dump(mode="json"),
        "ranking": (
            ranking.model_dump(mode="json")
            if ranking is not None
            else None
        ),
        "candidates": candidate_catalog,
        "same_compound_candidates": same_compound_candidates,
        "mechanistic_templates": mechanistic_templates,
        "top_same_compound_candidate": (
            same_compound_candidates[0]
            if same_compound_candidates else None
        ),
        "recommended_model_candidate": recommended_model_candidate,
        "recommended_use": recommended_use,
        "candidate_summary": {
            "retrieved": len(candidate_catalog),
            "catalog_scanned": len(catalog_candidates),
            "relevant_candidates": len(candidate_catalog),
            "excluded_unrelated": len(excluded_unrelated_models),
            "same_compound": len(same_compound_candidates),
            "mechanistic_template": len(mechanistic_templates),
            "reuse_ready": len(reuse_ready),
            "validated_reuse": len(
                candidates_by_layer["validated_reuse"]
            ),
            "controlled_adaptation": len(
                candidates_by_layer[
                    "controlled_adaptation_candidate"
                ]
            ),
            "same_compound_qualification": len(
                candidates_by_layer[
                    "same_compound_qualification_candidate"
                ]
            ),
            "mechanistic_reference": len(
                candidates_by_layer["mechanistic_reference"]
            ),
            "blocked": len(candidates_by_layer["blocked"]),
        },
        "candidates_by_layer": candidates_by_layer,
        "excluded_unrelated_models": excluded_unrelated_models,
        "relevance_audit": relevance_audit,
        "selected_model": selected,
        "abstention_reason_codes": reason_codes,
        "blueprint_status": (
            "not_required_model_reuse_selected"
            if decision == "reuse"
            else "same_compound_adaptation_required"
            if recommended_use == "controlled_adaptation"
            else "same_compound_qualification_required"
            if recommended_use == "qualification_required"
            else "eligible_for_template_lookup"
        ),
        "methodology": _methodology(),
    }


def _selected_model_details(
    session: Session,
    catalog: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    run = session.execute(
        text(
            """
            SELECT
                external_validation_run_id,
                validation_grade,
                run_label_grade,
                ranking_label,
                engineering_accepted,
                scientific_accepted,
                verification_dataset_count,
                pooled_observation_count,
                pooled_gmfe,
                pooled_geometric_bias,
                pooled_two_fold_fraction,
                endpoint_scenario_pass_fraction,
                audit_report_sha256,
                protocol_sha256,
                windows_manifest_sha256,
                metadata_json,
                validated_at
            FROM public_pbpk_model_validation_run
            WHERE model_version_id = CAST(:model_version_id AS uuid)
            ORDER BY run_label_grade DESC, validated_at DESC
            LIMIT 1
            """
        ),
        {"model_version_id": catalog["model_version_id"]},
    ).mappings().one()
    references = [
        dict(row)
        for row in session.execute(
            text(
                """
                SELECT DISTINCT
                    dataset.reference,
                    dataset.source_location
                FROM public_pbpk_model_validation_dataset dataset
                JOIN public_pbpk_model_validation_run run
                  ON run.validation_run_id = dataset.validation_run_id
                WHERE run.model_version_id = CAST(:model_version_id AS uuid)
                ORDER BY dataset.reference, dataset.source_location
                """
            ),
            {"model_version_id": catalog["model_version_id"]},
        ).mappings()
    ]
    output = dict(catalog)
    output["constraint_result"] = candidate
    output["validation_run"] = dict(run)
    output["validation_run"]["validation_independence"] = (
        (run.get("metadata_json") or {}).get(
            "validation_independence",
            "repository_preregistered_holdout",
        )
    )
    output["validation_run"].pop("metadata_json", None)
    output["validation_references"] = references
    return output


def _methodology() -> dict[str, Any]:
    matcher = serving_status()
    return {
        "service_version": PUBLIC_MODEL_SERVICE_VERSION,
        "catalog_loader_version": MODEL_CATALOG_LOADER_VERSION,
        "ranker_version": RANKER_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "decision_method": "hard_gate_plus_within_layer_model_matching",
        "candidate_retrieval_method": (
            "complete_catalog_scan_then_provenance_bound_relevance_gate"
        ),
        "relevance_policy_version": MODEL_RELEVANCE_POLICY_VERSION,
        "candidate_layer_policy": "pbpk-public-model-five-layer-policy-1.0.0",
        "llm_used_for_decision": False,
        "gnn_used_for_decision": bool(matcher.get("learned")),
        "learned_ranking_enabled": bool(matcher.get("learned")),
        "matching_method": matcher.get("method"),
        "matching_benchmark_level": matcher.get("benchmark_level"),
        "matching_serving_status": matcher,
        "matching_warning": (
            "A checksum-verified learned matcher is loaded for within-layer "
            "ranking; its benchmark and serving scope are reported separately "
            "and it cannot override hard constraints."
            if matcher.get("learned")
            else "No accepted learned public-model matcher checkpoint is loaded. "
            "The displayed cross-graph score is a deterministic, versioned "
            "fallback and is not a calibrated probability."
        ),
        "confidence_semantics": (
            "Deterministic gate certainty; not a calibrated probability of "
            "clinical prediction success."
        ),
        "validation_independence_semantics": (
            "Grade 3 currently means a preregistered repository holdout split "
            "with frozen thresholds and independent code audit. It is not an "
            "external replication performed by an unrelated laboratory."
        ),
        "validation_transferability_semantics": (
            "Model recommendation is separate from scenario qualification. "
            "Direct reuse requires a registered model-verification context "
            "matching categorical constraints and exact or unit-normalized "
            "numerically equivalent continuous inputs. Interpolation and "
            "extrapolation retain the model as an adaptation candidate and "
            "require a new simulation and validation."
        ),
        "limitation": (
            "A catalog version is not a candidate unless it contains the "
            "requested compound or has a provenance-bound shared disposition "
            "process in the accepted Model KG. Relevant models are never "
            "promoted to reuse when license, independent validation, or "
            "another hard constraint is unresolved."
        ),
    }
