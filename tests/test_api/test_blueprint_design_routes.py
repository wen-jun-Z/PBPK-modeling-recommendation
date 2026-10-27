"""In-process API checks: no server, database, model service or simulation."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pbpk_kb.api.blueprint_design_routes import router


def _payload():
    source = {
        "provenance_id": "c064c9f7-1933-4b5f-a1a4-ef549a3f6a14",
        "provenance_kind": "model_asset", "source_id": "fixture-anatomy",
        "source_location": "fixture:anatomy", "source_checksum": "a" * 64,
    }
    return {
        "requirement": {
            "compound_semantic_id": "compound:fixture", "name": "Fixture",
            "route": "oral", "dose": 1, "dose_unit": "mg",
            "targets": [{"target_id": "plasma", "matrix": "plasma", "quantity": "concentration"}],
        },
        "base": {
            "base_id": "fixture-base", "version": "1.0.0", "supported_populations": ["healthy_adult"],
            "compartments": [{"compartment_id": "body", "semantic_id": "organism:human",
                              "name": "Organism", "compartment_type": "other", "provenance": [source]}],
            "provenance": [source],
        },
    }


def test_stateless_design_returns_explicit_gaps_without_execution():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/blueprints/design", json=_payload())
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == "offline_design_only"
    assert result["candidates"][0]["status"] == "draft_with_gaps"
    assert result["execution"] == {
        "solver_calls": 0, "llm_calls": 0, "gpu_jobs": 0,
        "compile_status": "not_run", "scientific_status": "not_evaluated",
    }


def test_design_api_rejects_runtime_execution_flag():
    app = FastAPI()
    app.include_router(router)
    payload = _payload()
    payload["execute"] = True
    with TestClient(app) as client:
        response = client.post("/blueprints/design", json=payload)
    assert response.status_code == 422


def test_base_model_endpoint_returns_a_validated_pin_without_execution(monkeypatch):
    from pbpk_kb.api import blueprint_design_routes as routes
    definition = {
        "registry_id": "pksim-human-standard-small-molecule", "checksum": "a" * 64,
        "model": {"model_id": "4Comp", "species": "Human"},
        "status": "source_bound_standard_model_definition", "resources": [{"role": "database"}],
        "instantiation_requirements": ["Explicit individual and drug inputs"],
    }
    monkeypatch.setattr(routes, "load_standard_human_model", lambda root: definition)
    app = FastAPI()
    app.include_router(router)
    # Match the legacy UUID route shape to guard ordering of the fixed endpoint.
    @app.get("/blueprints/{blueprint_id}")
    def old_detail(blueprint_id: str):
        raise AssertionError("The fixed base-model route must take precedence")
    with TestClient(app) as client:
        response = client.get("/blueprints/base-model")
    assert response.status_code == 200
    body = response.json()
    assert body["human_model"]["definition_checksum"] == "a" * 64
    assert body["resource_validation"] == "passed"
    assert body["compile_status"] == body["simulation_status"] == "not_run"


def test_base_model_endpoint_reports_missing_resources(monkeypatch):
    from pbpk_kb.api import blueprint_design_routes as routes
    def missing(root):
        raise ValueError("MODEL_RESOURCE_CHECKSUM_MISMATCH")
    monkeypatch.setattr(routes, "load_standard_human_model", missing)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/blueprints/base-model")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "BASE_MODEL_RESOURCE_UNAVAILABLE"
