from pbpk_kb.storage.import_gnn_embeddings import _index_name, _sql_literal


def test_gnn_index_name_is_model_revision_isolated() -> None:
    first = _index_name("PBPK-HGT/model", "revision-a")
    second = _index_name("PBPK-HGT/model", "revision-b")
    assert first.startswith("recommendation_gnn_hnsw_")
    assert first.endswith("_idx")
    assert first != second


def test_gnn_sql_literal_escapes_quotes() -> None:
    assert _sql_literal("model's revision") == "'model''s revision'"
