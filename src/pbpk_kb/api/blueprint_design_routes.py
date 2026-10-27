"""Stateless design endpoint; never schedules model execution or GPU work."""
from pathlib import Path

from fastapi import APIRouter, HTTPException

from pbpk_kb.blueprints.design import design_blueprint
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from pbpk_kb.blueprints.human_model import load_standard_human_model

router = APIRouter(prefix="/blueprints", tags=["blueprints"])


@router.post("/design")
def create_design_draft(request: BlueprintDesignRequest) -> dict:
    return design_blueprint(request)


@router.get("/base-model")
def get_standard_human_model() -> dict:
    """Return a validated registry pin; does not create an individual or run a model."""
    try:
        definition = load_standard_human_model(Path(__file__).resolve().parents[3])
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail={"code": "BASE_MODEL_RESOURCE_UNAVAILABLE",
                                                    "detail": str(exc)}) from exc
    return {
        "human_model": {"registry_id": definition["registry_id"],
                        "definition_checksum": definition["checksum"]},
        "model": definition["model"], "status": definition["status"],
        "resource_validation": "passed", "resource_count": len(definition["resources"]),
        "instantiation_requirements": definition["instantiation_requirements"],
        "compile_status": "not_run", "simulation_status": "not_run",
    }
