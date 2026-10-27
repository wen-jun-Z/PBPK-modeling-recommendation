from fastapi import FastAPI

from pbpk_kb.api.compound_routes import router as compound_router
from pbpk_kb.api.evidence_routes import router as evidence_router
from pbpk_kb.api.export_routes import router as export_router
from pbpk_kb.api.graph_routes import router as graph_router
from pbpk_kb.api.model_routes import router as model_router
from pbpk_kb.api.qa_routes import router as qa_router
from pbpk_kb.api.recommendation_routes import router as recommendation_router
from pbpk_kb.api.blueprint_routes import (
    blueprint_router,
    job_router,
    simulation_router,
)
from pbpk_kb.api.blueprint_design_routes import router as blueprint_design_router
from pbpk_kb.api.search_routes import router as search_router
from pbpk_kb.api.source_routes import router as source_router
from pbpk_kb.config.settings import get_settings


settings = get_settings()

app = FastAPI(
    title="PBPK Stage 1 Public Knowledge Base API",
    version=settings.service_version,
)
app.include_router(search_router)
app.include_router(source_router)
app.include_router(compound_router)
app.include_router(model_router)
app.include_router(graph_router)
app.include_router(evidence_router)
app.include_router(qa_router)
app.include_router(recommendation_router)
app.include_router(blueprint_design_router)
app.include_router(blueprint_router)
app.include_router(job_router)
app.include_router(simulation_router)
app.include_router(export_router)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.service_name,
        "version": settings.service_version,
    }
