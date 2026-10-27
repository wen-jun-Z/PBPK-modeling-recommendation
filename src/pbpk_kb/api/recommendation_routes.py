from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from pbpk_kb.api.evidence_routes import serialize_row
from pbpk_kb.recommendation.gate6_evaluation import canonical_json
from pbpk_kb.recommendation.schemas import RecommendationRequest
from pbpk_kb.recommendation.public_model_service import (
    PublicModelRecommendationError,
    public_model_catalog_summary,
    public_model_matching_evaluation,
    recommend_public_model,
)
from pbpk_kb.config.settings import get_settings
from pbpk_kb.storage.db import get_db_session


router = APIRouter(prefix="/recommendation", tags=["recommendation"])


LATEST_ACCEPTED_RUN_SQL = """
SELECT *
FROM recommendation_model_run
WHERE engineering_acceptance IS TRUE
ORDER BY created_at DESC, recommendation_model_run_id DESC
LIMIT 1
"""


def _latest_run(session: Session) -> dict[str, Any]:
    try:
        row = session.execute(text(LATEST_ACCEPTED_RUN_SQL)).mappings().first()
    except ProgrammingError as exc:
        raise HTTPException(
            status_code=503,
            detail="Accepted recommendation artifacts have not been imported.",
        ) from exc
    if row is None:
        raise HTTPException(
            status_code=503,
            detail="No Gate 6 accepted recommendation model run is available.",
        )
    return serialize_row(row)


def _request_sha256(request: RecommendationRequest) -> str:
    payload = request.model_dump(mode="json")
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _knowledge_base_compounds(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT compound_id::text, preferred_name::text, pubchem_cid
            FROM compound_identity
            ORDER BY lower(preferred_name::text), compound_id
            """
        )
    ).mappings()
    return [serialize_row(row) for row in rows]


@router.post("/public-model-ranking")
def public_model_ranking(
    request: RecommendationRequest,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    try:
        return recommend_public_model(session=session, request=request)
    except PublicModelRecommendationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "recommendation_blocked",
                "errors": [str(exc)],
            },
        ) from exc


@router.get("/public-model-catalog-summary")
def model_catalog_summary(
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    return public_model_catalog_summary(session)


@router.get("/public-model-matching-evaluation")
def model_matching_evaluation() -> dict[str, Any]:
    return public_model_matching_evaluation()


@router.get("/compounds")
def recommendation_compounds(
    session: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    run = _latest_run(session)
    rows = session.execute(
        text(
            """
            WITH recommendation_coverage AS (
                SELECT compound_node_id,
                       COUNT(*)::integer AS validated_scenario_count,
                       array_agg(DISTINCT objective ORDER BY objective)
                           AS validated_objectives
                FROM recommendation_result
                WHERE recommendation_model_run_id = :run_id
                  AND status = 'recommendation_validated'
                GROUP BY compound_node_id
            ),
            evidence_coverage AS (
                SELECT attributed_compound_id,
                       COUNT(*)::integer AS auto_validated_evidence_count
                FROM auto_validated_evidence
                GROUP BY attributed_compound_id
            )
            SELECT compound.compound_id::text,
                   compound.preferred_name::text AS compound_name,
                   compound.pubchem_cid,
                   COALESCE(coverage.validated_scenario_count, 0)
                       AS validated_scenario_count,
                   COALESCE(
                       coverage.validated_objectives,
                       ARRAY[]::text[]
                   ) AS validated_objectives,
                   COALESCE(evidence.auto_validated_evidence_count, 0)
                       AS auto_validated_evidence_count
            FROM compound_identity compound
            LEFT JOIN recommendation_coverage coverage
              ON coverage.compound_node_id =
                 'Compound:' || compound.compound_id::text
            LEFT JOIN evidence_coverage evidence
              ON evidence.attributed_compound_id = compound.compound_id
            ORDER BY lower(compound.preferred_name::text), compound.compound_id
            """
        ),
        {"run_id": run["recommendation_model_run_id"]},
    ).mappings()
    return [serialize_row(row) for row in rows]


@router.get("/graph-summary")
def recommendation_graph_summary(
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    run = _latest_run(session)
    knowledge_base_compounds = _knowledge_base_compounds(session)
    benchmark_recommendation_count = session.execute(
        text(
            """
            SELECT COUNT(*)
            FROM recommendation_result
            WHERE recommendation_model_run_id = :run_id
              AND status = 'recommendation_validated'
            """
        ),
        {"run_id": run["recommendation_model_run_id"]},
    ).scalar_one()
    benchmark_compound_count = session.execute(
        text(
            """
            SELECT COUNT(DISTINCT compound_node_id)
            FROM recommendation_result
            WHERE recommendation_model_run_id = :run_id
              AND status = 'recommendation_validated'
            """
        ),
        {"run_id": run["recommendation_model_run_id"]},
    ).scalar_one()
    knowledge_base_recommendation_count = session.execute(
        text(
            """
            SELECT COUNT(*)
            FROM recommendation_result result
            JOIN compound_identity compound
              ON result.compound_node_id = 'Compound:' || compound.compound_id::text
            WHERE result.recommendation_model_run_id = :run_id
              AND result.status = 'recommendation_validated'
            """
        ),
        {"run_id": run["recommendation_model_run_id"]},
    ).scalar_one()
    covered_compound_count = session.execute(
        text(
            """
            SELECT COUNT(DISTINCT compound.compound_id)
            FROM recommendation_result result
            JOIN compound_identity compound
              ON result.compound_node_id = 'Compound:' || compound.compound_id::text
            WHERE result.recommendation_model_run_id = :run_id
              AND result.status = 'recommendation_validated'
            """
        ),
        {"run_id": run["recommendation_model_run_id"]},
    ).scalar_one()
    supported_objectives = [
        str(value)
        for value in session.execute(
            text(
                """
                SELECT DISTINCT objective
                FROM recommendation_result result
                JOIN compound_identity compound
                  ON result.compound_node_id =
                     'Compound:' || compound.compound_id::text
                WHERE result.recommendation_model_run_id = :run_id
                  AND result.status = 'recommendation_validated'
                ORDER BY objective
                """
            ),
            {"run_id": run["recommendation_model_run_id"]},
        ).scalars()
    ]
    metadata = run.get("metadata_json") or {}
    gate6 = metadata.get("gate6_manifest") or {}
    online_inference_enabled = os.getenv(
        "PBPK_RECOMMENDATION_ONLINE_ENABLED",
        "",
    ).lower() in {"1", "true", "yes"}
    knowledge_base_compound_count = len(knowledge_base_compounds)
    return {
        "recommendation_model_run_id": run["recommendation_model_run_id"],
        "validated_recommendation_count": int(
            knowledge_base_recommendation_count
        ),
        "compound_count": int(covered_compound_count),
        "knowledge_base_compound_count": knowledge_base_compound_count,
        "uncovered_compound_count": (
            knowledge_base_compound_count - int(covered_compound_count)
        ),
        "benchmark_validated_recommendation_count": int(
            benchmark_recommendation_count
        ),
        "benchmark_compound_count": int(benchmark_compound_count),
        "llm_model": run["llm_model"],
        "llm_revision": run["llm_revision"],
        "benchmark_level": run["benchmark_level"],
        "engineering_acceptance": bool(run["engineering_acceptance"]),
        "generalization_claim_eligible": bool(
            run["generalization_claim_eligible"]
        ),
        "gold_claim": bool(run["gold_claim"]),
        "supported_objectives": supported_objectives,
        "split_status": gate6.get("split_status") or {},
        "safety_acceptance": bool(gate6.get("safety_acceptance")),
        "online_inference_enabled": online_inference_enabled,
        "serving_mode": (
            "validated_results_and_online_inference"
            if online_inference_enabled
            else "validated_results_only"
        ),
    }


@router.get("/runtime-status")
def recommendation_runtime_status() -> dict[str, int]:
    from pbpk_kb.recommendation.online_agent import online_generation_status
    return online_generation_status()


@router.get("/validated")
def validated_recommendations(
    compound: str = Query(default=""),
    objective: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=500),
    session: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    run = _latest_run(session)
    rows = session.execute(
        text(
            """
            SELECT result.recommendation_result_id, result.example_id,
                   result.compound_node_id, result.compound_name,
                   result.objective, result.status, result.overall_confidence,
                   result.citation_count, result.evidence_gap_count,
                   result.request_json, result.created_at
            FROM recommendation_result result
            JOIN compound_identity compound
              ON result.compound_node_id = 'Compound:' || compound.compound_id::text
            WHERE result.recommendation_model_run_id = :run_id
              AND result.status = 'recommendation_validated'
              AND (:compound = '' OR lower(result.compound_name) = lower(:compound))
              AND (:objective = '' OR result.objective = :objective)
            ORDER BY lower(result.compound_name), result.objective,
                     result.example_id
            LIMIT :limit
            """
        ),
        {
            "run_id": run["recommendation_model_run_id"],
            "compound": compound.strip(),
            "objective": objective.strip(),
            "limit": limit,
        },
    ).mappings()
    return [serialize_row(row) for row in rows]


@router.get("/validated/{recommendation_result_id}")
def validated_recommendation_detail(
    recommendation_result_id: str,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    run = _latest_run(session)
    row = session.execute(
        text(
            """
            SELECT *
            FROM recommendation_result result
            WHERE result.recommendation_model_run_id = :run_id
              AND result.recommendation_result_id = :result_id
              AND result.status = 'recommendation_validated'
              AND EXISTS (
                  SELECT 1
                  FROM compound_identity compound
                  WHERE result.compound_node_id =
                        'Compound:' || compound.compound_id::text
              )
            """
        ),
        {
            "run_id": run["recommendation_model_run_id"],
            "result_id": recommendation_result_id,
        },
    ).mappings().first()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Validated recommendation result not found.",
        )
    value = serialize_row(row)
    value["model_run"] = {
        key: run[key]
        for key in (
            "recommendation_model_run_id",
            "llm_model",
            "llm_revision",
            "benchmark_level",
            "engineering_acceptance",
            "generalization_claim_eligible",
            "gold_claim",
        )
    }
    return value


@router.post("")
def recommend(
    request: RecommendationRequest,
    session: Session = Depends(get_db_session),
) -> dict[str, Any]:
    run = _latest_run(session)
    indexed_compound = session.execute(
        text(
            """
            SELECT compound_id::text, preferred_name::text
            FROM compound_identity
            WHERE lower(preferred_name::text) = lower(:compound)
            LIMIT 1
            """
        ),
        {"compound": request.compound.strip()},
    ).mappings().first()
    if indexed_compound is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Component recommendation is restricted to compounds indexed "
                "in the Stage 1 knowledge base."
            ),
        )
    request_hash = _request_sha256(request)
    row = session.execute(
        text(
            """
            SELECT *
            FROM recommendation_result result
            WHERE result.recommendation_model_run_id = :run_id
              AND result.request_sha256 = :request_sha256
              AND result.status = 'recommendation_validated'
              AND result.compound_node_id = :compound_node_id
            """
        ),
        {
            "run_id": run["recommendation_model_run_id"],
            "request_sha256": request_hash,
            "compound_node_id": "Compound:" + str(
                indexed_compound["compound_id"]
            ),
        },
    ).mappings().first()
    if row is None:
        if os.getenv(
            "PBPK_RECOMMENDATION_ONLINE_ENABLED",
            "",
        ).lower() in {"1", "true", "yes"}:
            from pbpk_kb.recommendation.online_agent import (
                configured_online_agent_kwargs,
                run_online_agent,
            )
            from pbpk_kb.storage.import_recommendation_runs import (
                import_online_recommendation_run,
            )

            try:
                online = run_online_agent(
                    request=request,
                    **configured_online_agent_kwargs(),
                )
            except (FileNotFoundError, LookupError, RuntimeError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "status": "recommendation_blocked",
                        "errors": [str(exc)],
                    },
                ) from exc
            if online["recommendation"] is None:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "status": online["validation"]["status"],
                        "errors": online["validation"].get("errors") or [],
                    },
                )
            imported = import_online_recommendation_run(
                run_dir=Path(online["run_dir"]),
                database_url=get_settings().database_url,
            )
            online["recommendation"]["recommendation_result_id"] = imported[
                "recommendation_result_id"
            ]
            online["recommendation"]["model_run"] = {
                "recommendation_model_run_id": run[
                    "recommendation_model_run_id"
                ],
                "llm_model": run["llm_model"],
                "llm_revision": run["llm_revision"],
                "benchmark_level": run["benchmark_level"],
                "engineering_acceptance": run[
                    "engineering_acceptance"
                ],
                "generalization_claim_eligible": run[
                    "generalization_claim_eligible"
                ],
                "gold_claim": run["gold_claim"],
            }
            return online["recommendation"]
        raise HTTPException(
            status_code=404,
            detail=(
                "No Gate 6 accepted recommendation exactly matches this "
                "ModelingRequest. Run the server-side online inference pipeline "
                "and validator before exposing a result."
            ),
        )
    value = serialize_row(row)
    recommendation = value["recommendation_json"]
    recommendation["recommendation_result_id"] = value[
        "recommendation_result_id"
    ]
    recommendation["model_run"] = {
        "recommendation_model_run_id": run["recommendation_model_run_id"],
        "llm_model": run["llm_model"],
        "llm_revision": run["llm_revision"],
        "benchmark_level": run["benchmark_level"],
        "engineering_acceptance": run["engineering_acceptance"],
        "generalization_claim_eligible": run[
            "generalization_claim_eligible"
        ],
        "gold_claim": run["gold_claim"],
    }
    return recommendation
