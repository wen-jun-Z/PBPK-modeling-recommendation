from __future__ import annotations

from pbpk_kb.recommendation.model_matching.repository_census import (
    classify_repository_tree,
)


def test_repository_tree_requires_license_model_data_and_report() -> None:
    result = classify_repository_tree(
        {
            "full_name": "Open-Systems-Pharmacology/Test-Model",
            "license_spdx_from_api": "GPL-2.0",
        },
        commit="a" * 40,
        license_texts={"LICENSE": "GNU GENERAL PUBLIC LICENSE Version 2"},
        paths=[
            "LICENSE",
            "Model/Test.pksim5",
            "Evaluation/Input/ObservedData.csv",
            "Evaluation/report.pdf",
            "Code/run.R",
        ],
    )

    assert result["static_processing_eligible"] is True
    assert result["execution_probe_eligible"] is True
    assert result["scientific_validation_candidate"] is True
    assert result["blockers"] == []


def test_repository_tree_does_not_treat_any_json_as_a_model() -> None:
    result = classify_repository_tree(
        {
            "full_name": "Open-Systems-Pharmacology/Metadata-Only",
            "license_spdx_from_api": "GPL-2.0",
        },
        commit="b" * 40,
        license_texts={"LICENSE": "GNU GENERAL PUBLIC LICENSE Version 2"},
        paths=["LICENSE", "package.json", "data/results.json"],
    )

    assert result["model_assets"] == []
    assert result["execution_probe_eligible"] is False
    assert "MODEL_ASSET_MISSING" in result["blockers"]


def test_readme_license_and_model_snapshot_are_execution_sources() -> None:
    result = classify_repository_tree(
        {
            "full_name": "Open-Systems-Pharmacology/Modern-Model",
            "license_spdx_from_api": None,
        },
        commit="c" * 40,
        paths=[
            "README.md",
            "Modern-Model.json",
            "Evaluation/Input/Content/Modeling_Strategy.md",
        ],
        license_texts={
            "README.md": "This model is distributed under the GPLv2 License"
        },
    )

    assert result["native_executable_model_assets"] == []
    assert result["reconstructable_snapshot_assets"] == ["Modern-Model.json"]
    assert result["execution_probe_eligible"] is True
    assert result["scientific_validation_candidate"] is True
