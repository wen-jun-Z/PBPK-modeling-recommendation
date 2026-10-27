from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass


EMBEDDING_DIMENSION = 1536
EMBEDDING_MODEL = "hashing-embedding-v0"


@dataclass
class TextChunk:
    text: str
    section_title: str | None
    paragraph_number: int
    token_count: int


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def hash_embedding(text: str, dimensions: int = EMBEDDING_DIMENSION) -> list[float]:
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


def chunk_text(text: str, *, section_title: str | None = None, max_tokens: int = 180) -> list[TextChunk]:
    tokens = tokenize(text)
    if not tokens:
        return []
    chunks: list[TextChunk] = []
    words = text.split()
    current: list[str] = []
    paragraph_number = 1
    for word in words:
        current.append(word)
        if len(current) >= max_tokens:
            chunk = " ".join(current)
            chunks.append(
                TextChunk(
                    text=chunk,
                    section_title=section_title,
                    paragraph_number=paragraph_number,
                    token_count=len(tokenize(chunk)),
                )
            )
            paragraph_number += 1
            current = []
    if current:
        chunk = " ".join(current)
        chunks.append(
            TextChunk(
                text=chunk,
                section_title=section_title,
                paragraph_number=paragraph_number,
                token_count=len(tokenize(chunk)),
            )
        )
    return chunks


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension")
    return sum(l * r for l, r in zip(left, right))
