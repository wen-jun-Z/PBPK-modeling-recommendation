from __future__ import annotations

from pathlib import Path

import pytest

from pbpk_kb.recommendation.model_matching.snapshot_protocol_planner import (
    SnapshotProtocolPlannerError,
    _snapshot_path,
    infer_partition,
)


def test_partition_inference_supports_reference_sections() -> None:
    result = infer_partition(
        {
            "Sections": [
                {"Reference": "model-building", "Title": "Model Building"},
                {"Reference": "model-verification", "Title": "Model Verification"},
            ],
            "Plots": {
                "AllPlots": [
                    {
                        "Project": "Drug",
                        "Simulation": "building simulation",
                        "SectionReference": "model-building",
                    },
                    {
                        "Project": "Drug",
                        "Simulation": "verification simulation",
                        "SectionReference": "model-verification",
                    },
                ]
            },
        },
        project_ids={"Drug"},
    )

    assert result["development_section_references"] == ["model-building"]
    assert result["verification_section_references"] == ["model-verification"]
    assert result["roles_by_simulation"] == {
        "building simulation": "development",
        "verification simulation": "verification",
    }


def test_partition_inference_refuses_ambiguous_concentration_profile_section() -> None:
    with pytest.raises(
        SnapshotProtocolPlannerError,
        match="do not explicitly identify building/verification",
    ):
        infer_partition(
            {
                "Plots": {
                    "AllPlots": [
                        {
                            "Project": "Drug",
                            "Simulation": "study",
                            "SectionReference": "ct-profiles",
                        }
                    ]
                }
            },
            project_ids={"Drug"},
        )


def test_snapshot_path_uses_unique_same_commit_basename_fallback(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "Evaluation" / "Input" / "evaluation_plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text("{}", encoding="utf-8")
    snapshot = tmp_path / "Drug-Model.json"
    snapshot.write_text("{}", encoding="utf-8")

    resolved, declared, mode = _snapshot_path(
        plan,
        {"Path": "../Drug-Model.json"},
        tmp_path,
    )

    assert resolved == snapshot.resolve()
    assert declared == "../Drug-Model.json"
    assert mode == "unique_basename_fallback"


def test_snapshot_path_refuses_external_dependency_without_lock(tmp_path: Path) -> None:
    plan = tmp_path / "evaluation_plan.json"
    plan.write_text("{}", encoding="utf-8")

    with pytest.raises(
        SnapshotProtocolPlannerError,
        match="EXTERNAL_SNAPSHOT_REFERENCE_REQUIRES_DEPENDENCY_LOCK",
    ):
        _snapshot_path(
            plan,
            {"Path": "https://example.test/Drug-Model.json"},
            tmp_path,
        )
