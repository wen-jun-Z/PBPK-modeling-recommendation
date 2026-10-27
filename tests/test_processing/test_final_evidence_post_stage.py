from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from pbpk_kb.processing.final_evidence_post_stage import (
    analyze_conflict_rows,
    audit_prior_stages,
    build_conflict_record,
    build_evidence_layers,
    normalize_stage_summary_paths,
)


def evidence(evidence_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "evidence_id": evidence_id,
        "evidence_type": "clinical_pk",
        "parameter": "auc",
        "canonical_value": 10.0,
        "canonical_value_low": None,
        "canonical_value_high": None,
        "canonical_unit": "ng*h/ml",
        "value": 10.0,
        "unit": "ng*h/ml",
        "value_text": "10",
        "route": None,
        "population": None,
        "condition": None,
        "dose": None,
        "source_text": "Caffeine AUC was 10 ng*h/mL.",
        "source_location": f"pubmed:test#chunk={evidence_id}",
        "source_id": "pubmed",
        "curation_status": "needs_review",
        "compound_role": "target_compound",
        "compound_id": "compound-caffeine",
        "compound_name": "caffeine",
        "attributed_compound_id": "compound-caffeine",
        "attributed_compound_name": "caffeine",
        "domain_gate_status": "passed",
        "metadata_json": {},
    }
    row.update(overrides)
    return row


def test_same_context_numeric_disagreement_is_unresolved_conflict() -> None:
    rows = [evidence("one"), evidence("two", canonical_value=20.0, value=20.0, value_text="20")]

    result = analyze_conflict_rows(rows)

    assert result["status"] == "unresolved_conflict"
    assert result["requires_human_review"] is True


def test_explicit_dose_difference_is_explainable_variation() -> None:
    rows = [
        evidence("one", dose="5 mg"),
        evidence("two", dose="10 mg", canonical_value=20.0, value=20.0, value_text="20"),
    ]

    result = analyze_conflict_rows(rows)

    assert result["status"] == "explainable_variation"
    assert result["requires_human_review"] is False
    assert result["explicit_context_differences"] == ["dose"]


def test_partial_context_with_disagreement_requires_review() -> None:
    rows = [
        evidence("one", condition=None),
        evidence("two", condition="fed", canonical_value=20.0, value=20.0, value_text="20"),
    ]

    result = analyze_conflict_rows(rows)

    assert result["status"] == "insufficient_context"
    assert result["partial_context_fields"] == ["condition"]


def test_overlapping_numeric_ranges_are_compatible() -> None:
    rows = [
        evidence("one", canonical_value=None, canonical_value_low=8.0, canonical_value_high=12.0),
        evidence("two", canonical_value=None, canonical_value_low=11.0, canonical_value_high=14.0),
    ]

    assert analyze_conflict_rows(rows)["status"] == "compatible"


def test_free_text_condition_paraphrases_are_not_automatically_explainable() -> None:
    rows = [
        evidence("one", condition="QuickStrip delivery versus oral gavage", canonical_value=None, value=None, value_text=None),
        evidence("two", condition="QuickStrip formulation compared with gavage", canonical_value=None, value=None, value_text=None),
    ]

    result = analyze_conflict_rows(rows)

    assert result["status"] == "insufficient_context"
    assert result["uncertain_context_fields"] == ["condition"]


def test_controlled_tags_inside_free_text_condition_remain_review_only() -> None:
    rows = [
        evidence("one", condition="fed administration", canonical_value=10.0, value=10.0),
        evidence("two", condition="fasted administration", canonical_value=20.0, value=20.0),
    ]

    result = analyze_conflict_rows(rows)

    assert result["status"] == "insufficient_context"
    assert result["condition_category_values"] == ["fasted", "fed"]


def test_compatible_semantic_conflict_group_still_requires_duplicate_review() -> None:
    rows = [evidence("one"), evidence("two")]
    analysis = analyze_conflict_rows(rows)

    record = build_conflict_record(
        conflict_group_id="group",
        source_group_type="semantic_conflict_candidate",
        source_group_id="semantic-group",
        rows=rows,
        analysis=analysis,
    )

    assert record["analysis_status"] == "compatible"
    assert record["requires_human_review"] is True
    assert record["resolution"] == "defer_to_human_duplicate_review"


def test_stage_summary_paths_are_normalized_relative_to_run(tmp_path) -> None:
    run = tmp_path / "run"
    state = {"stages": {"provenance": {"summary_path": str(run / "00_run_manifest" / "summary.json")}}}

    normalize_stage_summary_paths(state, run)

    assert state["stages"]["provenance"]["summary_path"] == "00_run_manifest/summary.json"


def test_evidence_layers_partition_candidates_and_preserve_review_blockers() -> None:
    candidates = [evidence(value) for value in ("reviewable", "det-duplicate", "similar", "domain-review", "conflict")]
    attributions = [
        {
            "evidence_id": row["evidence_id"],
            "compound_id": "compound-caffeine",
            "compound_name": "caffeine",
            "compound_role": "target_compound",
            "attribution_status": "resolved",
            "attribution_method": "rule",
            "attribution_stage": "rule",
            "attribution_confidence": 0.9,
            "attribution_reason": "test",
            "matched_text": "Caffeine",
        }
        for row in candidates
    ]
    domain = [
        {
            "evidence_id": row["evidence_id"],
            "domain_gate_status": "needs_manual_review" if row["evidence_id"] == "domain-review" else "passed",
            "domain_classification": "test",
            "reasons": ["direction"] if row["evidence_id"] == "domain-review" else [],
        }
        for row in candidates
    ]
    representatives = [deepcopy(row) for row in candidates if row["evidence_id"] != "det-duplicate"]
    inputs = {
        "candidates": candidates,
        "provenance": [{"evidence_id": row["evidence_id"], "provenance_status": "passed"} for row in candidates],
        "attributions": attributions,
        "domain_results": domain,
        "deterministic_members": [
            {
                "evidence_id": "det-duplicate",
                "duplicate_group_id": "det-group",
                "is_representative": False,
            }
        ],
        "deterministic_representatives": representatives,
        "semantic_members": [
            {
                "evidence_id": "similar",
                "duplicate_group_id": "similar-group",
                "group_type": "similar_for_review",
                "is_representative": True,
            }
        ],
    }
    conflict_report = [
        {
            "conflict_group_id": "conflict-group",
            "analysis_status": "unresolved_conflict",
            "requires_human_review": True,
        }
    ]
    conflict_members = [
        {
            "conflict_group_id": "conflict-group",
            "evidence_id": "conflict",
            "analysis_status": "unresolved_conflict",
            "requires_human_review": True,
        }
    ]

    outputs, summary = build_evidence_layers(inputs, conflict_report, conflict_members)

    assert [row["evidence_id"] for row in outputs["reviewable_evidence.jsonl"]] == ["reviewable"]
    assert {row["evidence_id"] for row in outputs["preselected_evidence.jsonl"]} == {"reviewable", "similar", "conflict"}
    assert {row["evidence_id"] for row in outputs["excluded_from_reviewable.jsonl"]} == {
        "det-duplicate",
        "similar",
        "domain-review",
        "conflict",
    }
    assert outputs["accepted_evidence.jsonl"] == []
    assert outputs["modeling_evidence.jsonl"] == []
    assert summary["accepted"] is True



def audited_run(tmp_path, candidate_count=2, identity_count=3):
    """A complete small accepted run, intentionally unlike historical counts."""
    source = tmp_path / "source.txt"
    text = "Caffeine AUC was 10 ng*h/mL."
    source.write_text(text)
    candidates = [
        evidence(
            str(index), run_id="new-run", raw_asset_id="raw",
            storage_path=str(source), checksum="source-checksum",
            source_text_hash=hashlib.sha256(text.encode()).hexdigest(),
            extraction_method="llm", extractor_name="local-qwen",
            extractor_version="test", parameter=f"auc-{index}",
        )
        for index in range(candidate_count)
    ]
    inputs = {
        "candidates": candidates,
        "provenance": [{"evidence_id": row["evidence_id"], "provenance_status": "passed"} for row in candidates],
        "attributions": [{"evidence_id": row["evidence_id"], "attribution_status": "resolved", "compound_id": "compound-caffeine"} for row in candidates],
        "identities": [{"compound_id": "compound-caffeine" if index == 0 else f"compound-{index}"} for index in range(identity_count)],
        "domain_results": [{"evidence_id": row["evidence_id"], "domain_gate_status": "passed", "reasons": []} for row in candidates],
        "deterministic_representatives": deepcopy(candidates),
        "deterministic_members": [],
        "semantic_input": deepcopy(candidates),
        "semantic_groups": [],
        "semantic_members": [],
        "semantic_state": [{"group_key": f"compound-caffeine|clinical_pk|auc-{index}", "status": "skipped_small"} for index in range(candidate_count)],
    }
    summaries = {
        "provenance": {"candidate_count": candidate_count},
        "compound_attribution": {"candidate_count": candidate_count, "compound_identity_count": identity_count},
        "domain_quality_gates": {"candidate_count": candidate_count, "passed_count": candidate_count, "needs_manual_review_count": 0, "excluded_count": 0},
        "deterministic_deduplication": {"representative_count": candidate_count, "semantic_input_count": candidate_count},
        "semantic_deduplication": {"input_rows": candidate_count, "candidate_groups_total": candidate_count},
    }
    state = {"stages": {}}
    for name, summary in summaries.items():
        summary["accepted"] = True
        relative = name + ".json"
        (tmp_path / relative).write_text(json.dumps(summary))
        state["stages"][name] = {"status": "completed", "accepted": True, "summary_path": relative}
    state["stages"]["semantic_deduplication"].update(
        embedding_model="NeuML/pubmedbert-base-embeddings", embedding_dimension=768,
    )
    return inputs, state, summaries


@pytest.mark.parametrize("candidate_count,identity_count", [(1, 1), (4, 3), (2, 11)])
def test_prior_stage_audit_uses_current_accepted_run_counts(tmp_path, candidate_count, identity_count):
    inputs, state, _ = audited_run(tmp_path, candidate_count, identity_count)
    audit = audit_prior_stages(inputs, state, tmp_path)
    assert audit["accepted"], audit["errors"]
    assert audit["candidate_count"] == candidate_count
    assert audit["compound_identity_count"] == identity_count


@pytest.mark.parametrize("stage,field", [
    ("provenance", "candidate_count"),
    ("compound_attribution", "candidate_count"),
    ("compound_attribution", "compound_identity_count"),
    ("domain_quality_gates", "passed_count"),
    ("domain_quality_gates", "needs_manual_review_count"),
    ("domain_quality_gates", "excluded_count"),
    ("deterministic_deduplication", "representative_count"),
    ("deterministic_deduplication", "semantic_input_count"),
    ("semantic_deduplication", "input_rows"),
    ("semantic_deduplication", "candidate_groups_total"),
])
@pytest.mark.parametrize("replacement", [None, True, -1, 999])
def test_prior_stage_audit_rejects_missing_invalid_or_mismatched_counts(tmp_path, stage, field, replacement):
    inputs, state, summaries = audited_run(tmp_path)
    if replacement is None:
        summaries[stage].pop(field)
    else:
        summaries[stage][field] = replacement
    (tmp_path / state["stages"][stage]["summary_path"]).write_text(json.dumps(summaries[stage]))
    audit = audit_prior_stages(inputs, state, tmp_path)
    assert not audit["accepted"]
    assert any(stage in error and field in error for error in audit["errors"])


def test_prior_stage_audit_rejects_unaccepted_summary_and_id_coverage_gap(tmp_path):
    inputs, state, summaries = audited_run(tmp_path)
    summaries["provenance"]["accepted"] = False
    (tmp_path / "provenance.json").write_text(json.dumps(summaries["provenance"]))
    inputs["attributions"].pop()
    inputs["semantic_state"][1]["group_key"] = inputs["semantic_state"][0]["group_key"]
    audit = audit_prior_stages(inputs, state, tmp_path)
    assert not audit["accepted"]
    assert "provenance summary is not accepted" in audit["errors"]
    assert "attributions evidence_id coverage differs from candidates" in audit["errors"]
    assert "semantic state contains missing or duplicate group keys" in audit["errors"]
