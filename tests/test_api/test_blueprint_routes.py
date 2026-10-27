from __future__ import annotations

from pbpk_kb.api.main import app


def test_blueprint_job_and_simulation_routes_are_registered() -> None:
    paths = app.openapi()["paths"]

    assert paths["/blueprints"]["post"]["tags"] == ["blueprints"]
    assert "/blueprints/{blueprint_id}/validate" in paths
    assert "/blueprints/{blueprint_id}/compile" in paths
    assert "/blueprints/{blueprint_id}/run" in paths
    assert "/blueprints/{blueprint_id}/package" in paths
    assert "/jobs/{job_id}/retry" in paths
    assert "/simulation-runs/{simulation_run_id}" in paths
