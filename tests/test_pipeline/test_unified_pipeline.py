from __future__ import annotations

import json
from pathlib import Path

import pytest

from pbpk_kb.pipeline_utils import PIPELINE_STAGES, PipelineConfig, UnifiedStage1Pipeline


def make_config(tmp_path: Path, *, import_database: bool = False) -> PipelineConfig:
    raw_run = tmp_path / "raw" / "compound_runs" / "run_test"
    raw_run.mkdir(parents=True)
    sources = tmp_path / "sources.yaml"
    sources.write_text("- source_id: pubmed\n  enabled: true\n", encoding="utf-8")
    return PipelineConfig(
        compounds=("midazolam", "caffeine"),
        raw_output_root=tmp_path / "raw",
        processing_output_root=tmp_path / "processing",
        source_registry_path=sources,
        raw_run_root=raw_run,
        import_database=import_database,
    )


def accepted_handlers(names: list[str], calls: list[str]) -> dict[str, object]:
    handlers = {}
    for name in names:
        def run(stage_name: str = name) -> dict[str, object]:
            calls.append(stage_name)
            return {"accepted": True, "stage": stage_name}

        handlers[name] = run
    return handlers


def test_unified_pipeline_executes_the_strategy_order_exactly(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    expected = [stage.name for stage in PIPELINE_STAGES if stage.name != "database_import"]
    calls: list[str] = []
    pipeline = UnifiedStage1Pipeline(
        config,
        handlers=accepted_handlers(expected, calls),  # type: ignore[arg-type]
    )

    result = pipeline.run()

    assert calls == expected
    assert result["status"] == "completed"
    assert list(result["stages"]) == expected
    assert all(result["stages"][name]["accepted"] is True for name in expected)


def test_llm_stage_classifies_before_limited_real_extraction(monkeypatch, tmp_path):
    from pbpk_kb.processing import run_llm_extraction as extraction

    config = make_config(tmp_path)
    config.llm_max_requests = 10
    pipeline = UnifiedStage1Pipeline(config)
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {"accepted": False, "pending_count": 20}

    monkeypatch.setattr(extraction, "run_llm_extraction", run)
    result = pipeline._run_llm_extract()
    assert calls[0]["classify_only"] is True
    assert calls[0]["llm_max_requests"] == 0
    assert calls[1]["classify_only"] is False
    assert calls[1]["llm_max_requests"] == 10
    assert result["accepted"] is False
    assert (pipeline.raw_processing_dir / "00_run_manifest/llm_classification_summary.json").is_file()


def test_unified_pipeline_stops_at_the_first_unaccepted_stage(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    expected = [stage.name for stage in PIPELINE_STAGES if stage.name != "database_import"]
    calls: list[str] = []
    handlers = accepted_handlers(expected, calls)

    def reject_parse() -> dict[str, object]:
        calls.append("parse")
        return {"accepted": False}

    handlers["parse"] = reject_parse
    pipeline = UnifiedStage1Pipeline(config, handlers=handlers)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="parse"):
        pipeline.run()

    assert calls == expected[: expected.index("parse") + 1]
    manifest = json.loads(pipeline.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["current_stage"] == "parse"
    assert manifest["stages"]["parse"]["accepted"] is False
    assert "deterministic_extract" not in manifest["stages"]


def test_unified_pipeline_resume_does_not_repeat_accepted_stages(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    expected = [stage.name for stage in PIPELINE_STAGES if stage.name != "database_import"]
    first_calls: list[str] = []
    first = UnifiedStage1Pipeline(
        config,
        handlers=accepted_handlers(expected, first_calls),  # type: ignore[arg-type]
    )
    first.run()

    repeated_calls: list[str] = []
    second = UnifiedStage1Pipeline(
        config,
        handlers=accepted_handlers(expected, repeated_calls),  # type: ignore[arg-type]
    )
    result = second.run()

    assert first_calls == expected
    assert repeated_calls == []
    assert result["status"] == "completed"


def test_plan_includes_database_only_when_enabled(tmp_path: Path) -> None:
    without_database = UnifiedStage1Pipeline(
        make_config(tmp_path / "without", import_database=False),
        handlers={},
    ).plan()
    with_database = UnifiedStage1Pipeline(
        make_config(tmp_path / "with", import_database=True),
        handlers={},
    ).plan()

    assert without_database["stages"][-1]["name"] == "llm_quality_validation"
    assert with_database["stages"][-1]["name"] == "database_import"
    assert [row["position"] for row in with_database["stages"]] == list(
        range(1, len(PIPELINE_STAGES) + 1)
    )


def test_existing_run_derives_its_actual_raw_output_root(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    pipeline = UnifiedStage1Pipeline(config, handlers={})

    assert pipeline.raw_output_root == config.raw_run_root.parents[1]
    assert pipeline.plan()["raw_output_root"] == str(config.raw_run_root.parents[1])


def test_latest_matching_incomplete_manifest_is_resumed_automatically(tmp_path: Path) -> None:
    first_config = make_config(tmp_path)
    expected = [stage.name for stage in PIPELINE_STAGES if stage.name != "database_import"]
    first_calls: list[str] = []
    first_handlers = accepted_handlers(expected, first_calls)

    def reject_parse() -> dict[str, object]:
        first_calls.append("parse")
        return {"accepted": False}

    first_handlers["parse"] = reject_parse
    first = UnifiedStage1Pipeline(first_config, handlers=first_handlers)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        first.run()

    resumed_config = PipelineConfig(
        compounds=first_config.compounds,
        raw_output_root=first_config.raw_output_root,
        processing_output_root=first_config.processing_output_root,
        source_registry_path=first_config.source_registry_path,
        raw_run_root=None,
        import_database=False,
    )
    resumed_calls: list[str] = []
    resumed = UnifiedStage1Pipeline(
        resumed_config,
        handlers=accepted_handlers(expected, resumed_calls),  # type: ignore[arg-type]
    )
    result = resumed.run()

    assert resumed.raw_run_root == first.raw_run_root
    assert resumed_calls == expected[expected.index("parse") :]
    assert result["status"] == "completed"


def test_explicit_empty_raw_directory_starts_new_acquisition(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config = make_config(tmp_path)
    pipeline = UnifiedStage1Pipeline(config)
    calls = []
    retriever = SimpleNamespace(
        bind_new_run_root=lambda path: calls.append(("bind", path)),
        run=lambda: calls.append(("run", None)) or {"fixture": True},
    )
    monkeypatch.setattr(pipeline, "_make_retriever", lambda: retriever)
    result = pipeline._run_raw_acquisition()
    assert result["accepted"] is True
    assert calls == [("bind", config.raw_run_root.resolve()), ("run", None)]


@pytest.mark.parametrize("method", ["_run_llm_extract", "_run_provenance_attribution", "_run_llm_quality_validation"])
def test_deferred_llm_check_blocks_each_unfinished_llm_stage(tmp_path, method):
    config = make_config(tmp_path)
    calls = []

    def unavailable():
        calls.append("checked")
        raise RuntimeError("fixture backend unavailable")

    config.llm_preflight = unavailable
    pipeline = UnifiedStage1Pipeline(config)
    with pytest.raises(RuntimeError, match="fixture backend unavailable"):
        getattr(pipeline, method)()
    assert calls == ["checked"]


def test_explicit_run_refuses_different_compound_set(tmp_path):
    config = make_config(tmp_path)
    names = [stage.name for stage in PIPELINE_STAGES if stage.name != "database_import"]
    pipeline = UnifiedStage1Pipeline(config, handlers=accepted_handlers(names, []))
    pipeline.run()
    config.compounds = ("other-drug",)
    with pytest.raises(RuntimeError, match="compounds do not match"):
        UnifiedStage1Pipeline(config).run()


def test_nonpositive_pmc_limit_is_rejected(tmp_path):
    config = make_config(tmp_path)
    config.max_pmc_fulltext_per_compound = 0
    with pytest.raises(ValueError, match="limits must be positive"):
        config.validate()
