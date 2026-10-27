from pbpk_kb.recommendation.model_catalog import (
    LATEST_VALIDATION_SQL,
    VALIDATION_METRIC_COUNT_SQL,
    VALIDATION_SCOPE_SQL,
)


def test_validation_scope_sql_is_unambiguous_and_run_bound() -> None:
    sql = " ".join(str(VALIDATION_SCOPE_SQL).split())

    for column in (
        "external_simulation_id",
        "simulation_name",
        "protocol_role",
        "species",
        "populations",
        "routes",
        "formulations",
        "validation_dataset_name",
    ):
        assert f"simulation.{column}" in sql
    assert "selected.selection_rank" not in sql
    assert "ranked.selection_rank = 1" in sql
    assert "dataset.validation_run_id = selected.validation_run_id" in sql
    assert (
        "dataset.simulation_sequence = simulation.protocol_sequence"
        in sql
    )


def test_catalog_batch_queries_select_one_validation_run_per_version() -> None:
    latest = " ".join(str(LATEST_VALIDATION_SQL).split())
    metrics = " ".join(str(VALIDATION_METRIC_COUNT_SQL).split())

    for sql in (latest, metrics):
        assert "PARTITION BY run.model_version_id" in sql
        assert "run.run_label_grade DESC" in sql
        assert "run.validated_at DESC" in sql
        assert "selection_rank = 1" in sql
    assert "metric.validation_run_id = selected.validation_run_id" in metrics
