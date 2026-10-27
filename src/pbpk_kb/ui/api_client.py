from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def api_base_url() -> str:
    return os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def request_json(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    timeout_seconds: int = 30,
) -> Any:
    filtered_params = {
        key: value
        for key, value in (params or {}).items()
        if value is not None and value != ""
    }
    query = f"?{urlencode(filtered_params)}" if filtered_params else ""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{api_base_url()}{path}{query}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body else None
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        try:
            detail = json.loads(error_body)
        except json.JSONDecodeError:
            detail = error_body
        raise RuntimeError(f"API request failed: {exc.code} {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"API service is unavailable: {exc}") from exc


def list_sources(*, family: str = "", enabled: bool | None = None, include_planned: bool = False) -> list[dict[str, Any]]:
    return request_json(
        "GET",
        "/sources",
        params={
            "family": family,
            "enabled": enabled,
            "include_planned": include_planned,
        },
    )


def update_source(
    *,
    source_id: str,
    enabled: bool | None = None,
    owner_curator: str | None = None,
    requested_by: str = "stage1_admin",
) -> dict[str, Any]:
    return request_json(
        "PATCH",
        f"/sources/{source_id}",
        payload={
            "enabled": enabled,
            "owner_curator": owner_curator,
            "requested_by": requested_by,
        },
    )


def list_ingestion_runs(
    *,
    source_id: str = "",
    status: str = "",
    limit: int = 50,
) -> list[dict[str, Any]]:
    return request_json(
        "GET",
        "/ingestion-runs",
        params={
            "source_id": source_id,
            "status": status,
            "limit": limit,
        },
    )


def ingestion_run_log(ingestion_run_id: str) -> dict[str, Any]:
    return request_json("GET", f"/ingestion-runs/{ingestion_run_id}/log")


def retry_ingestion_run(
    *,
    ingestion_run_id: str,
    query: str = "midazolam",
    retmax: int = 1,
    allow_network: bool = False,
    requested_by: str = "stage1_admin",
) -> dict[str, Any]:
    return request_json(
        "POST",
        f"/ingestion-runs/{ingestion_run_id}/retry",
        payload={
            "query": query,
            "retmax": retmax,
            "allow_network": allow_network,
            "requested_by": requested_by,
        },
    )


def ingest_source(
    *,
    source_id: str,
    query: str = "midazolam",
    retmax: int = 1,
    allow_network: bool = False,
    requested_by: str = "stage1_admin",
) -> dict[str, Any]:
    return request_json(
        "POST",
        f"/sources/{source_id}/ingest",
        params={
            "query": query,
            "retmax": retmax,
            "allow_network": allow_network,
            "requested_by": requested_by,
        },
    )


def source_ids() -> list[str]:
    return [row["source_id"] for row in list_sources()]


def source_families() -> list[str]:
    return sorted({row["source_family"] for row in list_sources() if row.get("source_family")})


def search_compounds(query: str, limit: int = 25) -> list[dict[str, Any]]:
    return request_json("GET", "/compounds/search", params={"q": query, "limit": limit})


def get_compound(compound_id: str) -> dict[str, Any]:
    return request_json("GET", f"/compounds/{compound_id}")


def search_models(
    *,
    compound: str = "",
    source_id: str = "",
    software: str = "",
    validation_status: str = "",
    reusable_only: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    return request_json(
        "GET",
        "/models/search",
        params={
            "compound": compound,
            "source_id": source_id,
            "software": software,
            "validation_status": validation_status,
            "reusable_only": reusable_only,
            "limit": limit,
        },
    )


def get_model_detail(model_id: str) -> dict[str, Any]:
    return request_json("GET", f"/models/{model_id}")


def search_evidence(
    *,
    compound_id: str = "",
    compound: str = "",
    evidence_type: str = "",
    source_id: str = "",
    source_family: str = "",
    species: str = "",
    population: str = "",
    route: str = "",
    dose: str = "",
    enzyme_or_transporter: str = "",
    source_date_from: str = "",
    source_date_to: str = "",
    curation_status: str = "",
    min_confidence: float = 0.0,
    q: str = "",
    limit: int = 100,
) -> list[dict[str, Any]]:
    return request_json(
        "GET",
        "/evidence/search",
        params={
            "compound_id": compound_id,
            "compound": compound,
            "evidence_type": evidence_type,
            "source_id": source_id,
            "source_family": source_family,
            "species": species,
            "population": population,
            "route": route,
            "dose": dose,
            "enzyme_or_transporter": enzyme_or_transporter,
            "source_date_from": source_date_from,
            "source_date_to": source_date_to,
            "curation_status": curation_status,
            "min_confidence": min_confidence,
            "q": q,
            "limit": limit,
        },
    )


def semantic_search(query: str, source_id: str = "", limit: int = 25) -> list[dict[str, Any]]:
    return request_json("GET", "/search/semantic", params={"q": query, "source_id": source_id, "limit": limit})


def graph_summary() -> dict[str, Any]:
    return request_json("GET", "/graph/summary")


def recommendation_compounds() -> list[dict[str, str]]:
    return request_json("GET", "/recommendation/compounds")


def recommendation_graph_summary() -> dict[str, Any]:
    return request_json("GET", "/recommendation/graph-summary")


def validated_recommendations(
    compound: str = "",
    objective: str = "",
    limit: int = 100,
) -> list[dict[str, Any]]:
    return request_json(
        "GET",
        "/recommendation/validated",
        params={
            "compound": compound,
            "objective": objective,
            "limit": limit,
        },
    )


def validated_recommendation_detail(
    recommendation_result_id: str,
) -> dict[str, Any]:
    return request_json(
        "GET",
        f"/recommendation/validated/{recommendation_result_id}",
    )


def recommend_pbpk(
    *,
    compound: str,
    objective: str,
    route: str = "",
    population: str = "",
    interacting_drug: str = "",
    constraints: str = "",
    top_k_subgraphs: int = 8,
    max_evidence_per_component: int = 5,
) -> dict[str, Any]:
    return recommend_pbpk_request(
        {
            "compound": compound,
            "objective": objective,
            "route": route or None,
            "population": population or None,
            "interacting_drug": interacting_drug or None,
            "constraints": constraints or None,
            "top_k_subgraphs": top_k_subgraphs,
            "max_evidence_per_component": max_evidence_per_component,
        }
    )


def recommend_pbpk_request(payload: dict[str, Any]) -> dict[str, Any]:
    return request_json(
        "POST",
        "/recommendation",
        payload=payload,
        timeout_seconds=1800,
    )


def rank_public_model(payload: dict[str, Any]) -> dict[str, Any]:
    return request_json(
        "POST",
        "/recommendation/public-model-ranking",
        payload=payload,
        timeout_seconds=120,
    )


def public_model_catalog_summary() -> dict[str, Any]:
    return request_json(
        "GET",
        "/recommendation/public-model-catalog-summary",
    )


def public_model_matching_evaluation() -> dict[str, Any]:
    return request_json(
        "GET",
        "/recommendation/public-model-matching-evaluation",
    )


def create_blueprint(payload: dict[str, Any]) -> dict[str, Any]:
    return request_json(
        "POST",
        "/blueprints",
        payload=payload,
        timeout_seconds=180,
    )


def list_blueprints(compound: str = "") -> list[dict[str, Any]]:
    return request_json("GET", "/blueprints", params={"compound": compound})


def blueprint_detail(blueprint_id: str) -> dict[str, Any]:
    return request_json("GET", f"/blueprints/{blueprint_id}")


def blueprint_versions(blueprint_id: str) -> list[dict[str, Any]]:
    return request_json("GET", f"/blueprints/{blueprint_id}/versions")


def blueprint_parameters(blueprint_id: str) -> list[dict[str, Any]]:
    return request_json("GET", f"/blueprints/{blueprint_id}/parameters")


def blueprint_components(
    blueprint_id: str,
    component_kind: str = "",
) -> list[dict[str, Any]]:
    return request_json(
        "GET",
        f"/blueprints/{blueprint_id}/components",
        params={"component_kind": component_kind, "limit": 2000},
    )


def blueprint_artifacts(blueprint_id: str) -> list[dict[str, Any]]:
    return request_json("GET", f"/blueprints/{blueprint_id}/artifacts")


def validate_blueprint(blueprint_id: str) -> dict[str, Any]:
    return request_json("POST", f"/blueprints/{blueprint_id}/validate")


def compile_blueprint(blueprint_id: str) -> dict[str, Any]:
    return request_json("POST", f"/blueprints/{blueprint_id}/compile")


def run_blueprint(blueprint_id: str) -> dict[str, Any]:
    return request_json("POST", f"/blueprints/{blueprint_id}/run")


def blueprint_simulation_runs(blueprint_id: str) -> list[dict[str, Any]]:
    return request_json(
        "GET",
        f"/blueprints/{blueprint_id}/simulation-runs",
    )


def blueprint_package_url(blueprint_id: str) -> str:
    return f"{api_base_url()}/blueprints/{blueprint_id}/package"


def blueprint_package_bytes(blueprint_id: str) -> bytes:
    """Fetch internally so remote browsers never need a localhost API URL."""
    try:
        with urlopen(blueprint_package_url(blueprint_id), timeout=120) as response:
            return response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("Blueprint package download is unavailable.") from exc


def compound_graph(compound_id: str) -> dict[str, Any]:
    return request_json("GET", f"/graph/compound/{compound_id}")


def graph_centers(center_type: str) -> list[dict[str, Any]]:
    return request_json("GET", f"/graph/centers/{center_type}")


def target_graph(target_id: str) -> dict[str, Any]:
    return request_json("GET", f"/graph/target/{target_id}")


def source_graph(source_id: str) -> dict[str, Any]:
    return request_json("GET", f"/graph/source/{source_id}")


def model_graph(model_id: str) -> dict[str, Any]:
    return request_json("GET", f"/graph/model/{model_id}")


def review_queue(
    *,
    status: str = "needs_review",
    evidence_type: str = "",
    source_id: str = "",
    compound: str = "",
    q: str = "",
    min_confidence: float | None = None,
    created_from: str = "",
    created_to: str = "",
    duplicate_risk: str = "all",
    missing_unit: bool | None = None,
    missing_context: bool | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    return request_json(
        "GET",
        "/evidence/review-queue",
        params={
            "status": status,
            "evidence_type": evidence_type,
            "source_id": source_id,
            "compound": compound,
            "q": q,
            "min_confidence": min_confidence,
            "created_from": created_from,
            "created_to": created_to,
            "duplicate_risk": duplicate_risk,
            "missing_unit": missing_unit,
            "missing_context": missing_context,
            "limit": limit,
        },
    )


def evidence_detail(evidence_type: str, evidence_id: str) -> dict[str, Any]:
    return request_json("GET", f"/evidence/{evidence_type}/{evidence_id}")


def curate_evidence(
    *,
    evidence_type: str,
    evidence_id: str,
    action: str,
    curator: str,
    comment: str | None = None,
    field_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return request_json(
        "POST",
        f"/evidence/{evidence_type}/{evidence_id}/curate",
        payload={
            "action": action,
            "curator": curator,
            "comment": comment,
            "field_updates": field_updates or {},
        },
    )


def ask_question(
    *,
    question: str,
    compound: str = "",
    evidence_limit: int = 8,
    chunk_limit: int = 5,
    model_limit: int = 5,
) -> dict[str, Any]:
    return request_json(
        "POST",
        "/qa/ask",
        payload={
            "question": question,
            "compound": compound or None,
            "evidence_limit": evidence_limit,
            "chunk_limit": chunk_limit,
            "model_limit": model_limit,
        },
    )


def export_evidence(**payload: Any) -> dict[str, Any]:
    return request_json("POST", "/exports/evidence", payload=payload)


def export_compound_profile(**payload: Any) -> dict[str, Any]:
    return request_json("POST", "/exports/compound-profile", payload=payload)


def export_model_registry(**payload: Any) -> dict[str, Any]:
    return request_json("POST", "/exports/model-registry", payload=payload)


def export_stage1_summary(**payload: Any) -> dict[str, Any]:
    return request_json("POST", "/exports/stage1-summary", payload=payload)


def export_jobs(limit: int = 25) -> list[dict[str, Any]]:
    return request_json("GET", "/exports/jobs", params={"limit": limit})
