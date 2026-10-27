import json
import math
from pathlib import Path

import pytest

from pbpk_kb.processing.semantic_dedup_file import (
    FINAL_DEDUPLICATION_DIR,
    SemanticDedupRunner,
    build_conflict_record,
    build_group_record,
    build_member_record,
    classify_high_similarity_cluster,
    conflicting_fields,
    has_value_or_condition_conflict,
    remaining_review_indexes,
    run_relative_path,
    select_representative,
    semantic_text,
)


def evidence(evidence_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "evidence_id": evidence_id,
        "evidence_type": "clinical_pk",
        "parameter": "AUC",
        "source_text": "Caffeine AUC was measured after oral dosing in healthy adults.",
        "source_id": "pubmed",
        "source_location": f"pubmed:test#chunk={evidence_id}",
        "source_text_hash": f"hash-{evidence_id}",
        "attributed_compound_id": "compound-caffeine",
        "attributed_compound_name": "caffeine",
        "provenance_status": "passed",
        "attribution_confidence": 0.9,
        "extraction_confidence": 0.8,
        "value": None,
        "unit": None,
        "canonical_value": None,
        "canonical_unit": None,
        "condition": None,
        "population": None,
        "route": None,
        "dose": None,
    }
    row.update(overrides)
    return row


def test_semantic_text_stably_includes_attributed_compound_and_evidence_fields() -> None:
    text = semantic_text(evidence("one"), 1500)

    assert text.splitlines()[:3] == ["caffeine", "clinical_pk", "AUC"]
    assert "Caffeine AUC was measured" in text


@pytest.mark.parametrize("field,left,right,label", [
    ("value_high", 2.0, 3.0, "range_unit"),
    ("canonical_value_low", 0.0, 1.0, "canonical_range_unit"),
    ("species", ["rat"], ["human"], "species"),
    ("value", 0.0, 1.0, "value_unit"),
])
def test_semantic_dedup_does_not_merge_distinct_ranges_species_or_zero(
    field: str, left: object, right: object, label: str,
) -> None:
    rows = [evidence("one", **{field: left}), evidence("two", **{field: right})]
    assert has_value_or_condition_conflict(rows)
    assert label in conflicting_fields(rows)


def test_conflict_record_lists_scientifically_relevant_conflicting_fields() -> None:
    rows = [
        evidence("one", value=10.0, unit="mg h/L", condition="fasted"),
        evidence("two", value=20.0, unit="mg h/L", condition="fed"),
    ]
    group = build_group_record(
        duplicate_group_id="group",
        group_type="semantic_conflict_candidate",
        group_key="key",
        rows=rows,
        representative=rows[0],
        model_name="NeuML/pubmedbert-base-embeddings",
        device="cuda",
        threshold=0.92,
        review_threshold=0.85,
    )

    record = build_conflict_record(group, rows)

    assert conflicting_fields(rows) == ["value_unit", "condition"]
    assert record["group_type"] == "semantic_conflict_candidate"
    assert record["resolution"] == "defer_to_conflict_analysis_and_human_review"


def test_semantic_representative_uses_deterministic_strategy_priority() -> None:
    lower = evidence("lower", attribution_confidence=0.7, extraction_confidence=0.99)
    higher = evidence("higher", attribution_confidence=0.95, extraction_confidence=0.70)

    assert select_representative([lower, higher])["evidence_id"] == "higher"


def test_review_layer_excludes_high_similarity_duplicates_and_conflict_members() -> None:
    assert remaining_review_indexes(7, {1, 2, 4, 5}) == [0, 3, 6]


def test_high_similarity_cluster_requires_all_pairs_for_automatic_duplicate() -> None:
    class VectorMath:
        @staticmethod
        def dot(left: list[float], right: list[float]) -> float:
            return sum(a * b for a, b in zip(left, right, strict=True))

    offset = math.sqrt(1.0 - 0.95**2)
    embeddings = [[1.0, 0.0], [0.95, offset], [0.95, -offset]]
    rows = [evidence("one"), evidence("two"), evidence("three")]

    assert classify_high_similarity_cluster(rows[:2], embeddings, [0, 1], 0.92, VectorMath) == "semantic_duplicate"
    assert classify_high_similarity_cluster(rows, embeddings, [0, 1, 2], 0.92, VectorMath) == "similar_for_review"
    rows[1]["condition"] = "fed"
    rows[2]["condition"] = "fasted"
    assert classify_high_similarity_cluster(rows, embeddings, [0, 1, 2], 0.92, VectorMath) == "semantic_conflict_candidate"


def test_member_record_retains_group_type() -> None:
    row = evidence("one")
    record = build_member_record("group", "similar_for_review", row, row, 1, 1.0)

    assert record["group_type"] == "similar_for_review"


def test_run_relative_path_does_not_persist_execution_host_prefix(tmp_path: Path) -> None:
    output = tmp_path / "run"
    path = output / "03_deduplication" / "summary.json"

    assert run_relative_path(path, output) == "03_deduplication/summary.json"


def test_runner_requires_domain_and_deterministic_stages_before_embedding(tmp_path: Path) -> None:
    output = tmp_path / "final"
    input_path = output / FINAL_DEDUPLICATION_DIR / "semantic_dedup_input.jsonl"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("", encoding="utf-8")
    manifest = output / "00_run_manifest" / "pipeline_state.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"stages": {"semantic_deduplication": {"status": "not_started"}}}), encoding="utf-8")
    runner = SemanticDedupRunner(
        input_path=input_path,
        output_dir=output,
        model_name="model",
        device="cpu",
        threshold=0.92,
        review_threshold=0.85,
        batch_size=2,
        max_source_chars=1000,
        min_source_chars=40,
        min_group_size=2,
        max_group_size=100,
        reset=False,
    )

    with pytest.raises(RuntimeError, match="domain_quality_gates"):
        runner._validate_pipeline_state()



@pytest.mark.parametrize("field,left,right", [
    ("value_qualifier", "<", ">"),
    ("value_qualifier", None, ">"),
    ("value_text", "increased", "decreased"),
    ("value_text", None, "increased"),
    ("species", ["rat"], None),
])
def test_high_similarity_does_not_erase_qualifiers_text_or_unknown_species(field, left, right):
    class VectorMath:
        @staticmethod
        def dot(left, right):
            return sum(a * b for a, b in zip(left, right))
    rows = [evidence("one", **{field: left}), evidence("two", **{field: right})]
    assert has_value_or_condition_conflict(rows)
    assert field in conflicting_fields(rows)
    assert classify_high_similarity_cluster(
        rows, [[1.0, 0.0], [1.0, 0.0]], [0, 1], 0.92, VectorMath,
    ) == "semantic_conflict_candidate"


@pytest.mark.parametrize("fields", [
    {},
    {"value": 10.0, "unit": "%", "value_qualifier": "<", "value_text": "less than 10", "species": ["rat"]},
    {"value_text": "increased", "species": None},
])
def test_identical_fact_still_deduplicates_with_qualifiers_text_and_species(fields):
    class VectorMath:
        @staticmethod
        def dot(left, right):
            return sum(a * b for a, b in zip(left, right))
    rows = [evidence("one", **fields), evidence("two", **fields)]
    assert not has_value_or_condition_conflict(rows)
    assert conflicting_fields(rows) == []
    assert classify_high_similarity_cluster(
        rows, [[1.0, 0.0], [1.0, 0.0]], [0, 1], 0.92, VectorMath,
    ) == "semantic_duplicate"
