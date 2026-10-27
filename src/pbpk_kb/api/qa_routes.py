from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from pbpk_kb.qa.answer_generator import generate_answer
from pbpk_kb.qa.retrieval import retrieve_for_question
from pbpk_kb.storage.db import get_db_session


router = APIRouter(tags=["qa"])


class QARequest(BaseModel):
    question: str = Field(min_length=3)
    compound: str | None = None
    evidence_limit: int = Field(default=8, ge=1, le=25)
    chunk_limit: int = Field(default=5, ge=0, le=20)
    model_limit: int = Field(default=5, ge=0, le=20)


@router.post("/qa/ask")
def ask_question(request: QARequest, db: Session = Depends(get_db_session)) -> dict[str, Any]:
    retrieval = retrieve_for_question(
        db,
        question=request.question,
        compound=request.compound,
        evidence_limit=request.evidence_limit,
        chunk_limit=request.chunk_limit,
        model_limit=request.model_limit,
    )
    return generate_answer(request.question, retrieval)
