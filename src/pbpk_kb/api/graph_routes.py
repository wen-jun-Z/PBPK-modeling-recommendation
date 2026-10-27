from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from pbpk_kb.graph.graph_queries import compound_graph as compound_graph_query
from pbpk_kb.graph.graph_queries import (
    graph_centers,
    graph_summary,
    model_graph as model_graph_query,
    source_graph as source_graph_query,
    target_graph as target_graph_query,
)
from pbpk_kb.storage.db import get_db_session


router = APIRouter(tags=["graph"])


@router.get("/graph/compound/{compound_id}")
def compound_graph(compound_id: str, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    graph = compound_graph_query(db, compound_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Compound not found")
    return graph


@router.get("/graph/summary")
def graph_summary_route(db: Session = Depends(get_db_session)) -> dict[str, Any]:
    return graph_summary(db)


@router.get("/graph/centers/{center_type}")
def list_graph_centers(center_type: str, db: Session = Depends(get_db_session)) -> list[dict[str, Any]]:
    try:
        return graph_centers(db, center_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/graph/target/{target_id}")
def target_graph(target_id: str, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    graph = target_graph_query(db, target_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Protein target not found")
    return graph


@router.get("/graph/source/{source_id}")
def source_graph(source_id: str, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    graph = source_graph_query(db, source_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return graph


@router.get("/graph/model/{model_id}")
def model_graph(model_id: str, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    graph = model_graph_query(db, model_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Public model not found")
    return graph
