from pathlib import Path

from pbpk_kb.paths import (
    configured_path,
    model_matching_root,
    processing_output_root,
    raw_output_root,
    recommendation_model_cache_dir,
    recommendation_quality_dir,
    resolve_recorded_path,
)


def test_configured_path_uses_default_when_environment_is_empty(monkeypatch) -> None:
    monkeypatch.setenv("PBPK_TEST_PATH", "")

    assert configured_path("PBPK_TEST_PATH", Path("fallback")) == Path("fallback")


def test_runtime_paths_honor_environment_overrides(monkeypatch, tmp_path) -> None:
    values = {
        "PBPK_RAW_OUTPUT_ROOT": tmp_path / "raw",
        "PBPK_PROCESSING_OUTPUT_ROOT": tmp_path / "processed",
        "PBPK_RECOMMENDATION_QUALITY_DIR": tmp_path / "quality",
        "PBPK_RECOMMENDATION_MODEL_CACHE_DIR": tmp_path / "cache",
        "PBPK_MODEL_MATCHING_ROOT": tmp_path / "model-matching",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))

    assert raw_output_root() == values["PBPK_RAW_OUTPUT_ROOT"]
    assert processing_output_root() == values["PBPK_PROCESSING_OUTPUT_ROOT"]
    assert recommendation_quality_dir() == values["PBPK_RECOMMENDATION_QUALITY_DIR"]
    assert recommendation_model_cache_dir() == values["PBPK_RECOMMENDATION_MODEL_CACHE_DIR"]
    assert model_matching_root() == values["PBPK_MODEL_MATCHING_ROOT"]


def test_resolve_recorded_path_maps_legacy_server_prefix() -> None:
    resolved = resolve_recorded_path(
        "/root/autodl-tmp/pbpk-model-assets/model_matching/run/manifest.json"
    )

    assert resolved == model_matching_root() / "run" / "manifest.json"


def test_resolve_recorded_path_maps_legacy_sync_and_external_prefixes() -> None:
    sync_path = resolve_recorded_path(
        "/root/autodl-tmp/pbpk_stage1_public_kb_ui_plan_updated_sync/data_process/run.json"
    )
    external_path = resolve_recorded_path(
        "/root/autodl-tmp/external/MolCLR/LICENSE"
    )

    assert sync_path.parts[-2:] == ("data_process", "run.json")
    assert external_path.parts[-3:] == ("external", "MolCLR", "LICENSE")
