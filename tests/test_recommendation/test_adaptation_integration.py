from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.recommendation.model_matching.adaptation_integration import (
    integrate_controlled_adaptation_audits,
)
from pbpk_kb.recommendation.model_matching.execution_benchmark import (
    build_automatic_labels_from_audits,
)


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_passed_adaptation_becomes_grade_two_with_real_supervision(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "benchmark"
    task = {
        "task_id": "task-1",
        "validation_group_id": "group-1",
        "active_moieties": ["drug"],
        "required_model_compounds": ["drug"],
        "split": "train",
    }
    candidate = {
        "candidate_id": "candidate-1",
        "active_moieties": ["drug"],
        "model_family": "Drug",
    }
    pair = {
        "task_id": "task-1",
        "validation_group_id": "group-1",
        "candidate_id": "candidate-1",
        "relevance_grade": 1,
        "decision": "mechanistic_reference",
        "label_origin": "mechanistic_reference",
        "same_active_moiety": True,
        "exact_executed_scenario": False,
        "adaptation_required": True,
        "adaptation_operation_count": 0,
        "reason_codes": ["CONTROLLED_ADAPTATION_EXECUTION_REQUIRED"],
    }
    checksums = {
        "benchmark_tasks.jsonl": _write_jsonl(
            benchmark / "benchmark_tasks.jsonl", [task]
        ),
        "benchmark_truth.jsonl": _write_jsonl(
            benchmark / "benchmark_truth.jsonl", [{"task_id": "task-1"}]
        ),
        "candidates.jsonl": _write_jsonl(
            benchmark / "candidates.jsonl", [candidate]
        ),
        "candidate_pairs.jsonl": _write_jsonl(
            benchmark / "candidate_pairs.jsonl", [pair]
        ),
        "leakage_audit.json": _write_json(
            benchmark / "leakage_audit.json",
            {
                "passed": True,
                "task_assignments": [
                    {"task_id": "task-1", "split": "train"}
                ],
                "candidate_assignments": [
                    {"candidate_id": "candidate-1", "split": "train"}
                ],
            },
        ),
    }
    _write_json(
        benchmark / "manifest.json",
        {
            "benchmark_version": "test",
            "benchmark_build_version": "test",
            "scientific_claim_eligible": True,
            "scientific_claim_blockers": [],
            "output_checksums": checksums,
        },
    )

    adaptations = tmp_path / "adaptations"
    plan = {
        "adaptation_plan_id": "adaptation-plan:one",
        "protocol_key": "one",
        "task_id": "task-1",
        "candidate_id": "candidate-1",
        "adaptation_operation_multilabel": ["dose"],
        "adaptation_operation_count": 1,
        "adaptation_weighted_cost": 1.0,
        "execution_adjustments": [
            {
                "adjustment_type": "output_horizon",
                "new_value": 24.25,
                "unit": "h",
            }
        ],
        "execution_adjustment_count": 1,
    }
    adaptation_checksums = {
        "adaptation_plans.jsonl": _write_jsonl(
            adaptations / "adaptation_plans.jsonl", [plan]
        ),
        "blocked_adaptations.jsonl": _write_jsonl(
            adaptations / "blocked_adaptations.jsonl", []
        ),
        "protocols.jsonl": _write_jsonl(
            adaptations / "protocols.jsonl", [{"protocol_key": "one"}]
        ),
    }
    _write_json(
        adaptations / "manifest.json",
        {"output_checksums": adaptation_checksums},
    )

    windows = tmp_path / "windows"
    _write_json(
        windows / "one/windows_run_manifest.json",
        {
            "runs": [
                {"simulations": [{"duration_seconds": 2.5}]},
                {"simulations": [{"duration_seconds": 3.5}]},
            ]
        },
    )
    audits = tmp_path / "audits"
    audit_row = {
        "protocol_key": "one",
        "audit_status": "audited_passed",
        "engineering_accepted": True,
        "scientific_accepted": True,
        "audit_report_sha256": "a" * 64,
    }
    audit_index_sha = _write_jsonl(audits / "audit_index.jsonl", [audit_row])
    failures_sha = _write_jsonl(audits / "failures.jsonl", [])
    _write_json(
        audits / "summary.json",
        {
            "windows_results_root": str(windows),
            "counts": {
                "runner_failed_count": 0,
                "auditor_failed_count": 0,
            },
            "output_checksums": {
                "audit_index.jsonl": audit_index_sha,
                "failures.jsonl": failures_sha,
            },
        },
    )

    integrated = tmp_path / "integrated"
    result = integrate_controlled_adaptation_audits(
        benchmark_directory=benchmark,
        adaptation_directory=adaptations,
        adaptation_audit_directory=audits,
        output_directory=integrated,
    )

    assert result["controlled_adaptation"]["executed_count"] == 1
    assert result["controlled_adaptation"]["scientific_passed_count"] == 1
    assert result["decision_grade_counts"] == {
        "0": 0,
        "1": 0,
        "2": 1,
        "3": 0,
    }
    assert result["label_counts"] == result["decision_grade_counts"]
    assert result["output_checksums"]["leakage_audit.json"] == checksums[
        "leakage_audit.json"
    ]
    assert result["abstention_task_count"] == 0
    assert result["abstention_task_fraction"] == 0.0
    assert result["hard_negative_complete_task_count"] == 0
    assert result["soft_negative_complete_task_count"] == 0
    assert (
        "DIFFICULT_SOFT_NEGATIVE_QUERY_COVERAGE_INCOMPLETE"
        in result["scientific_claim_blockers"]
    )
    integrated_pair = json.loads(
        (integrated / "candidate_pairs.jsonl").read_text(encoding="utf-8")
    )
    assert integrated_pair["relevance_grade"] == 2
    assert integrated_pair["label_origin"] == "controlled_adaptation_execution"
    assert integrated_pair["adaptation_executed"] is True
    assert integrated_pair["adaptation_operation_multilabel"] == ["dose"]
    assert integrated_pair["adaptation_runtime_seconds"] == 6.0
    assert integrated_pair["adaptation_execution_adjustment_count"] == 1
    assert integrated_pair["adaptation_execution_adjustments"][0][
        "adjustment_type"
    ] == "output_horizon"

    labels = tmp_path / "labels"
    label_manifest = build_automatic_labels_from_audits(
        benchmark_directory=integrated,
        output_directory=labels,
    )
    assert label_manifest["adaptation_supervision_available"] is True
    label = json.loads((labels / "labels.jsonl").read_text(encoding="utf-8"))
    assert label["adaptation_weighted_cost"] == 1.0
    assert label["adaptation_scientific_passed"] is True
