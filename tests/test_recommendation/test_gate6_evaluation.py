from __future__ import annotations

import pytest

from pbpk_kb.recommendation.gate6_evaluation import (
    build_split_manifest,
    build_time_manifest,
    set_metrics,
)


def silver_row(
    example_id: str,
    compound_id: str,
    split: str,
    document: str,
    *,
    model: str | None = None,
) -> dict:
    return {
        "example_id": example_id,
        "compound_id": compound_id,
        "split": split,
        "source_document_ids": [document],
        "public_model_checksums": [model] if model else [],
        "target": {
            "evidence_ids": [f"evidence:{example_id}"],
            "public_model_ids": [f"model-node:{model}"] if model else [],
        },
    }


def test_set_metrics_treats_empty_empty_as_specificity_not_false_f1() -> None:
    report = set_metrics(
        [set(), {"E01"}, {"E02"}],
        [set(), {"E01"}, {"E01"}],
    )

    assert report["true_positive"] == 1
    assert report["false_positive"] == 1
    assert report["false_negative"] == 1
    assert report["f1"] == pytest.approx(0.5)
    assert report["exact_set_accuracy"] == pytest.approx(2 / 3)
    assert report["empty_target_specificity"] == 1.0


def test_split_manifest_detects_resource_leakage() -> None:
    rows = [
        silver_row("train-a", "compound:a", "train", "shared-document"),
        silver_row("test-b", "compound:b", "test", "shared-document"),
        silver_row("test-c", "compound:c", "test", "test-document"),
    ]

    report = build_split_manifest(
        axis="source_document",
        silver_rows=rows,
    )

    assert report["status"] == "insufficient_silver"
    assert report["resource_cross_split_count"] == 1


def test_public_model_axis_requires_two_test_compounds() -> None:
    rows = [
        silver_row(
            "test-a",
            "compound:a",
            "test",
            "doc-a",
            model="checksum-a",
        ),
        silver_row(
            "test-b",
            "compound:b",
            "test",
            "doc-b",
            model="checksum-b",
        ),
    ]

    report = build_split_manifest(axis="public_model", silver_rows=rows)

    assert report["status"] == "eligible"
    assert report["compound_counts"]["test"] == 2
    assert report["test_positive_target_count"] == 2


def test_time_split_does_not_substitute_download_dates() -> None:
    rows = [silver_row("test-a", "compound:a", "test", "doc-a")]

    report, ineligible = build_time_manifest(rows)

    assert report["status"] == "time_split_ineligible"
    assert report["download_date_used"] is False
    assert ineligible[0]["example_id"] == "test-a"
