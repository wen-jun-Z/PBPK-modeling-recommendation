from pbpk_kb.storage.vector_index import EMBEDDING_DIMENSION, chunk_text, cosine_similarity, hash_embedding


def test_hash_embedding_is_deterministic_and_normalized() -> None:
    first = hash_embedding("midazolam pharmacokinetics absorption")
    second = hash_embedding("midazolam pharmacokinetics absorption")

    assert first == second
    assert len(first) == EMBEDDING_DIMENSION
    assert cosine_similarity(first, second) > 0.99


def test_chunk_text_preserves_section_metadata() -> None:
    chunks = chunk_text(" ".join(["midazolam"] * 205), section_title="Pharmacokinetics", max_tokens=100)

    assert len(chunks) == 3
    assert chunks[0].section_title == "Pharmacokinetics"
    assert chunks[0].paragraph_number == 1
    assert chunks[0].token_count == 100
