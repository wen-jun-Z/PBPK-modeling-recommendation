from fastapi.testclient import TestClient

from pbpk_kb.api.main import app


def test_health_check() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "pbpk-stage1-public-kb",
        "version": "0.1.0",
    }


def test_stage1_planned_api_routes_are_registered() -> None:
    route_paths = set(app.openapi()["paths"])

    for path in [
        "/sources",
        "/sources/{source_id}",
        "/sources/{source_id}",
        "/sources/{source_id}/ingest",
        "/ingestion-runs",
        "/ingestion-runs/{ingestion_run_id}/log",
        "/ingestion-runs/{ingestion_run_id}/retry",
        "/compounds/{compound_id}",
        "/models/search",
        "/models/{model_id}",
        "/graph/summary",
        "/graph/compound/{compound_id}",
        "/recommendation",
        "/recommendation/compounds",
        "/recommendation/graph-summary",
        "/recommendation/validated",
        "/recommendation/validated/{recommendation_result_id}",
        "/recommendation/public-model-ranking",
        "/recommendation/public-model-catalog-summary",
        "/recommendation/public-model-matching-evaluation",
    ]:
        assert path in route_paths


def test_search_route_limits_cover_streamlit_usage() -> None:
    openapi = app.openapi()

    expected_maximums = {
        "/compounds/search": 250,
        "/evidence/search": 500,
        "/search/semantic": 100,
    }

    for path, expected_maximum in expected_maximums.items():
        parameters = openapi["paths"][path]["get"]["parameters"]
        limit_parameter = next(parameter for parameter in parameters if parameter["name"] == "limit")

        assert limit_parameter["schema"]["maximum"] == expected_maximum
