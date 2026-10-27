from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_NAME = os.getenv("PBPK_EMBEDDING_MODEL", "NeuML/pubmedbert-base-embeddings")
DEVICE = os.getenv("PBPK_EMBEDDING_DEVICE", "cuda")
EXPECTED_DIMENSION = 768

app = FastAPI(title="PBPK Biomedical Embedding Service", version="1.0")


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)
    normalize: bool = True


@lru_cache(maxsize=1)
def embedding_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required by the embedding service") from exc
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    dimension = int(model.get_sentence_embedding_dimension() or 0)
    if dimension != EXPECTED_DIMENSION:
        raise RuntimeError(f"Expected {EXPECTED_DIMENSION} dimensions, got {dimension}")
    return model


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        model = embedding_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Embedding model unavailable: {type(exc).__name__}") from exc
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "dimension": int(model.get_sentence_embedding_dimension()),
    }


@app.post("/embed")
def embed(request: EmbeddingRequest) -> dict[str, Any]:
    texts = [text.strip() for text in request.texts]
    if any(not text for text in texts):
        raise HTTPException(status_code=422, detail="Embedding text must not be empty")
    model = embedding_model()
    vectors = model.encode(
        texts,
        batch_size=min(64, len(texts)),
        normalize_embeddings=request.normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return {
        "model": MODEL_NAME,
        "device": DEVICE,
        "dimension": EXPECTED_DIMENSION,
        "normalized": request.normalize,
        "embeddings": vectors.astype("float32").tolist(),
    }
