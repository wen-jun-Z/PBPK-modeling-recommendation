from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from pbpk_kb.model_assets.license_policy import sha256_file


ADAPTATION_INTEGRATION_VERSION = "pbpk-controlled-adaptation-integration-1.3.0"


class AdaptationIntegrationError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AdaptationIntegrationError(
                    f"Expected JSON object at {path}:{line_number}."
                )
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=str)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = _canonical_bytes(row)
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _verify_output(
    root: Path, manifest: Mapping[str, Any], name: str
) -> Path:
    path = root / name
    expected = str((manifest.get("output_checksums") or {}).get(name) or "")
    if not expected or not path.is_file() or sha256_file(path) != expected:
        raise AdaptationIntegrationError(f"Checksum verification failed for {path}.")
    return path


def _runtime_seconds(windows_result_root: Path, protocol_key: str) -> float:
    manifest_path = windows_result_root / protocol_key / "windows_run_manifest.json"
    if not manifest_path.is_file():
        return 0.0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    return sum(
        float(simulation.get("duration_seconds") or 0.0)
        for run in manifest.get("runs") or []
        for simulation in run.get("simulations") or []
    )


def _replace_reason(
    reasons: object, *, replacement: str | None
) -> list[str]:
    values = {
        str(value)
        for value in (reasons or [])
        if str(value) != "CONTROLLED_ADAPTATION_EXECUTION_REQUIRED"
    }
    if replacement:
        values.add(replacement)
    return sorted(values)


def integrate_controlled_adaptation_audits(
    *,
    benchmark_directory: Path,
    adaptation_directory: Path,
    adaptation_audit_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    benchmark_directory = benchmark_directory.resolve()
    adaptation_directory = adaptation_directory.resolve()
    adaptation_audit_directory = adaptation_audit_directory.resolve()
    benchmark_manifest = json.loads(
        (benchmark_directory / "manifest.json").read_text(encoding="utf-8")
    )
    adaptation_manifest = json.loads(
        (adaptation_directory / "manifest.json").read_text(encoding="utf-8")
    )
    audit_summary = json.loads(
        (adaptation_audit_directory / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    pair_path = benchmark_directory / "candidate_pairs.jsonl"
    if sha256_file(pair_path) != str(
        benchmark_manifest["output_checksums"]["candidate_pairs.jsonl"]
    ):
        raise AdaptationIntegrationError("Benchmark candidate pairs checksum drifted.")
    plan_path = _verify_output(
        adaptation_directory, adaptation_manifest, "adaptation_plans.jsonl"
    )
    blocked_path = _verify_output(
        adaptation_directory, adaptation_manifest, "blocked_adaptations.jsonl"
    )
    audit_index_path = _verify_output(
        adaptation_audit_directory, audit_summary, "audit_index.jsonl"
    )
    plans = {
        str(row["protocol_key"]): row for row in _read_jsonl(plan_path)
    }
    blocked = {
        (str(row["task_id"]), str(row["candidate_id"])): row
        for row in _read_jsonl(blocked_path)
    }
    audit_rows = {
        str(row["protocol_key"]): row for row in _read_jsonl(audit_index_path)
    }
    if set(plans) != set(audit_rows):
        raise AdaptationIntegrationError(
            "Adaptation plans and audit rows do not have identical protocol keys."
        )
    if any(row.get("audit_status") == "auditor_failed" for row in audit_rows.values()):
        raise AdaptationIntegrationError(
            "Auditor failures cannot be integrated into scientific labels."
        )
    plan_by_pair = {
        (str(row["task_id"]), str(row["candidate_id"])): row
        for row in plans.values()
    }
    if set(plan_by_pair) & set(blocked):
        raise AdaptationIntegrationError("A pair is both planned and blocked.")
    windows_result_root = Path(str(audit_summary["windows_results_root"]))
    pairs = _read_jsonl(pair_path)
    updated: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for original in pairs:
        row = dict(original)
        pair_key = (str(row["task_id"]), str(row["candidate_id"]))
        if pair_key in blocked:
            blocked_row = blocked[pair_key]
            row.update(
                {
                    "adaptation_plan_id": blocked_row["adaptation_plan_id"],
                    "adaptation_attempt_status": "blocked",
                    "adaptation_executed": False,
                    "adaptation_scientific_passed": False,
                    "adaptation_operation_multilabel": [],
                    "adaptation_operation_count": 0,
                    "adaptation_weighted_cost": None,
                    "adaptation_execution_adjustments": [],
                    "adaptation_execution_adjustment_count": 0,
                    "adaptation_runtime_seconds": None,
                    "adaptation_failure_class": "planning_blocked",
                    "reason_codes": _replace_reason(
                        row.get("reason_codes"),
                        replacement="CONTROLLED_ADAPTATION_NOT_EXECUTABLE",
                    ),
                }
            )
            updated.append(row)
            continue
        plan = plan_by_pair.get(pair_key)
        if plan is None:
            updated.append(row)
            continue
        audit = audit_rows[str(plan["protocol_key"])]
        status = str(audit["audit_status"])
        engineering = bool(audit.get("engineering_accepted"))
        scientific = bool(audit.get("scientific_accepted"))
        runtime_seconds = _runtime_seconds(
            windows_result_root, str(plan["protocol_key"])
        )
        if status == "audited_passed" and engineering and scientific:
            row.update(
                {
                    "relevance_grade": 2,
                    "decision": "controlled_adaptation",
                    "label_origin": "controlled_adaptation_execution",
                    "adaptation_attempt_status": "audited_passed",
                    "reason_codes": _replace_reason(
                        row.get("reason_codes"), replacement=None
                    ),
                }
            )
            failure_class = None
        elif status == "audited_negative" and engineering and not scientific:
            row.update(
                {
                    "relevance_grade": 1,
                    "decision": "mechanistic_reference",
                    "label_origin": "controlled_adaptation_execution",
                    "adaptation_attempt_status": "audited_negative",
                    "reason_codes": _replace_reason(
                        row.get("reason_codes"),
                        replacement=(
                            "CONTROLLED_ADAPTATION_SCIENTIFIC_VALIDATION_FAILED"
                        ),
                    ),
                }
            )
            failure_class = "scientific_validation_failed"
        elif status == "runner_failed" and not engineering:
            row.update(
                {
                    "adaptation_attempt_status": "runner_failed",
                    "reason_codes": _replace_reason(
                        row.get("reason_codes"),
                        replacement="CONTROLLED_ADAPTATION_RUNNER_FAILED_NO_LABEL",
                    ),
                }
            )
            failure_class = "runner_failed"
        else:
            raise AdaptationIntegrationError(
                f"Inconsistent adaptation audit state for {plan['protocol_key']}: "
                f"{status}, engineering={engineering}, scientific={scientific}."
            )
        executed = engineering
        row.update(
            {
                "adaptation_plan_id": plan["adaptation_plan_id"],
                "adaptation_executed": executed,
                "adaptation_scientific_passed": bool(executed and scientific),
                "adaptation_operation_multilabel": (
                    plan["adaptation_operation_multilabel"] if executed else []
                ),
                "adaptation_operation_count": (
                    int(plan["adaptation_operation_count"]) if executed else 0
                ),
                "adaptation_weighted_cost": (
                    float(plan["adaptation_weighted_cost"]) if executed else None
                ),
                "adaptation_execution_adjustments": (
                    plan.get("execution_adjustments") or [] if executed else []
                ),
                "adaptation_execution_adjustment_count": (
                    int(plan.get("execution_adjustment_count") or 0)
                    if executed
                    else 0
                ),
                "adaptation_runtime_seconds": (
                    runtime_seconds if executed else None
                ),
                "adaptation_failure_class": failure_class,
                "adaptation_audit_report_sha256": audit.get(
                    "audit_report_sha256"
                ),
            }
        )
        outcomes.append(
            {
                "task_id": row["task_id"],
                "candidate_id": row["candidate_id"],
                "adaptation_plan_id": plan["adaptation_plan_id"],
                "protocol_key": plan["protocol_key"],
                "audit_status": status,
                "engineering_accepted": engineering,
                "scientific_accepted": scientific,
                "relevance_grade": row["relevance_grade"],
                "adaptation_operation_multilabel": row[
                    "adaptation_operation_multilabel"
                ],
                "adaptation_operation_count": row[
                    "adaptation_operation_count"
                ],
                "adaptation_weighted_cost": row["adaptation_weighted_cost"],
                "adaptation_execution_adjustments": row[
                    "adaptation_execution_adjustments"
                ],
                "adaptation_execution_adjustment_count": row[
                    "adaptation_execution_adjustment_count"
                ],
                "adaptation_runtime_seconds": row[
                    "adaptation_runtime_seconds"
                ],
                "adaptation_failure_class": failure_class,
                "audit_report_sha256": audit.get("audit_report_sha256"),
            }
        )
        updated.append(row)

    if len(updated) != len(pairs):
        raise AdaptationIntegrationError("Candidate-pair cardinality changed.")
    identities = {
        (str(row["task_id"]), str(row["candidate_id"])) for row in updated
    }
    if len(identities) != len(updated):
        raise AdaptationIntegrationError("Integrated candidate pairs are not unique.")
    outputs: dict[str, str] = {}
    for name in (
        "benchmark_tasks.jsonl",
        "benchmark_truth.jsonl",
        "candidates.jsonl",
        "leakage_audit.json",
    ):
        source = benchmark_directory / name
        expected = str(benchmark_manifest["output_checksums"][name])
        if sha256_file(source) != expected:
            raise AdaptationIntegrationError(f"Benchmark input drifted: {name}.")
        destination = output_directory / name
        shutil.copy2(source, destination)
        outputs[name] = sha256_file(destination)
    outputs["candidate_pairs.jsonl"] = _atomic_jsonl(
        output_directory / "candidate_pairs.jsonl", updated
    )
    outcomes.sort(
        key=lambda row: (str(row["task_id"]), str(row["candidate_id"]))
    )
    outputs["adaptation_results.jsonl"] = _atomic_jsonl(
        output_directory / "adaptation_results.jsonl", outcomes
    )
    grade_counts = Counter(int(row["relevance_grade"]) for row in updated)
    pairs_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in updated:
        pairs_by_task.setdefault(str(row["task_id"]), []).append(row)
    task_count = len(pairs_by_task)
    abstention_task_count = sum(
        not any(int(row["relevance_grade"]) >= 2 for row in rows)
        for rows in pairs_by_task.values()
    )
    abstention_fraction = (
        abstention_task_count / task_count if task_count else 0.0
    )
    hard_negative_complete_count = sum(
        any(int(row["relevance_grade"]) == 0 for row in rows)
        for rows in pairs_by_task.values()
    )
    soft_negative_complete_count = sum(
        any(int(row["relevance_grade"]) == 1 for row in rows)
        for rows in pairs_by_task.values()
    )
    blockers = set(benchmark_manifest.get("scientific_claim_blockers") or [])
    blockers.discard("CONTROLLED_ADAPTATION_EXECUTION_REQUIRED")
    for blocker in (
        "HARD_NEGATIVE_QUERY_COVERAGE_INCOMPLETE",
        "DIFFICULT_SOFT_NEGATIVE_QUERY_COVERAGE_INCOMPLETE",
        "ABSTENTION_TASK_FRACTION_LT_0_20",
        "REUSE_ADAPT_REFERENCE_REJECT_COVERAGE_INCOMPLETE",
    ):
        blockers.discard(blocker)
    if hard_negative_complete_count != task_count:
        blockers.add("HARD_NEGATIVE_QUERY_COVERAGE_INCOMPLETE")
    if soft_negative_complete_count != task_count:
        blockers.add("DIFFICULT_SOFT_NEGATIVE_QUERY_COVERAGE_INCOMPLETE")
    if abstention_fraction < 0.20:
        blockers.add("ABSTENTION_TASK_FRACTION_LT_0_20")
    audit_counts = audit_summary.get("counts") or {}
    if int(audit_counts.get("runner_failed_count") or 0) > 0:
        blockers.add("CONTROLLED_ADAPTATION_RUNNER_FAILURES_PRESENT")
    if int(audit_counts.get("auditor_failed_count") or 0) > 0:
        blockers.add("CONTROLLED_ADAPTATION_AUDITOR_FAILURES_PRESENT")
    if grade_counts.get(2, 0) == 0:
        blockers.add("CONTROLLED_ADAPTATION_GRADE_2_COUNT_ZERO")
    if any(grade_counts.get(grade, 0) == 0 for grade in range(4)):
        blockers.add("REUSE_ADAPT_REFERENCE_REJECT_COVERAGE_INCOMPLETE")
    manifest = {
        **benchmark_manifest,
        "benchmark_build_version": ADAPTATION_INTEGRATION_VERSION,
        "scientific_claim_eligible": not blockers,
        "scientific_claim_blockers": sorted(blockers),
        "candidate_pair_count": len(updated),
        "decision_grade_counts": {
            str(grade): grade_counts.get(grade, 0) for grade in range(4)
        },
        "label_counts": {
            str(grade): grade_counts.get(grade, 0) for grade in range(4)
        },
        "abstention_task_count": abstention_task_count,
        "abstention_task_fraction": abstention_fraction,
        "hard_negative_complete_task_count": hard_negative_complete_count,
        "soft_negative_complete_task_count": soft_negative_complete_count,
        "controlled_adaptation": {
            "planned_count": len(plans),
            "blocked_count": len(blocked),
            "executed_count": sum(
                bool(row.get("adaptation_executed")) for row in updated
            ),
            "scientific_passed_count": sum(
                bool(row.get("adaptation_scientific_passed")) for row in updated
            ),
            "runner_failed_count": int(
                audit_counts.get("runner_failed_count") or 0
            ),
            "adaptation_supervision_available": any(
                bool(row.get("adaptation_executed")) for row in updated
            ),
        },
        "source_benchmark_manifest_sha256": sha256_file(
            benchmark_directory / "manifest.json"
        ),
        "adaptation_manifest_sha256": sha256_file(
            adaptation_directory / "manifest.json"
        ),
        "adaptation_audit_summary_sha256": sha256_file(
            adaptation_audit_directory / "summary.json"
        ),
        "output_checksums": outputs,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(output_directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate audited controlled-adaptation executions into an immutable "
            "execution-derived recommendation benchmark."
        )
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--adaptations", type=Path, required=True)
    parser.add_argument("--adaptation-audits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = integrate_controlled_adaptation_audits(
        benchmark_directory=args.benchmark,
        adaptation_directory=args.adaptations,
        adaptation_audit_directory=args.adaptation_audits,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
