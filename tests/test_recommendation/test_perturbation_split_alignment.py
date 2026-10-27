from __future__ import annotations

import json

from pbpk_kb.recommendation.model_matching.perturbation_split_alignment import (
    align_perturbation_splits,
)


def test_perturbation_split_alignment_changes_only_split(tmp_path) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    base = {
        "task_id": "base-a",
        "split": "validation",
        "active_moieties": ["alfentanil"],
    }
    (benchmark / "benchmark_tasks.jsonl").write_text(
        json.dumps(base) + "\n", encoding="utf-8"
    )
    source = tmp_path / "perturbed_tasks.jsonl"
    perturbation = {
        "task_id": "perturbed-a",
        "base_task_id": "base-a",
        "split": "test",
        "perturbation_type": "relative_plus_1pct",
        "scenario": {"dose_value": 10.1, "dose_unit": "mg"},
    }
    source.write_text(json.dumps(perturbation) + "\n", encoding="utf-8")

    result = align_perturbation_splits(
        benchmark_directory=benchmark,
        perturbed_tasks_path=source,
        output_directory=tmp_path / "aligned",
    )

    aligned = json.loads(
        (tmp_path / "aligned/perturbed_tasks.jsonl").read_text()
    )
    assert aligned == {**perturbation, "split": "validation"}
    assert result["split_reassignment_count"] == 1
    assert result["perturbed_task_count"] == 1
    assert result["non_split_fields_preserved"] is True
