from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pbpk_kb.config.settings import get_settings


EXPECTED_DIMENSION = 768


class EmbeddingServiceError(RuntimeError):
    pass


def embed_query(text: str) -> list[float]:
    settings = get_settings()
    endpoint = (settings.embedding_api_url or "").rstrip("/")
    if not endpoint:
        raise EmbeddingServiceError("PBPK_EMBEDDING_API_URL is not configured")
    payload = json.dumps({"texts": [text], "normalize": True}).encode("utf-8")
    request = Request(
        f"{endpoint}/embed",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=settings.embedding_timeout_seconds) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise EmbeddingServiceError(f"Biomedical embedding service request failed: {type(exc).__name__}") from exc
    if result.get("model") != settings.embedding_model:
        raise EmbeddingServiceError("Biomedical embedding model mismatch")
    if result.get("dimension") != EXPECTED_DIMENSION:
        raise EmbeddingServiceError("Biomedical embedding dimension mismatch")
    vectors = result.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != 1 or not isinstance(vectors[0], list):
        raise EmbeddingServiceError("Biomedical embedding response is malformed")
    vector = [float(value) for value in vectors[0]]
    if len(vector) != EXPECTED_DIMENSION:
        raise EmbeddingServiceError("Biomedical embedding vector length mismatch")
    return vector
