from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.licensed_snapshot_reuse import (
    audit_snapshot_reuse_execution,
)


AUDIT_BATCH_VERSION = "pbpk-snapshot-execution-audit-batch-1.2.0"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, default=str)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        for row in rows:
            handle.write(
                (
                    json.dumps(
                        row,
                        ensure_ascii=True,
                        sort_keys=True,
                        default=str,
                    )
                    + "\n"
                ).encode("utf-8")
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _canonicalize_audit_report(
    *,
    audit_root: Path,
    protocol_id: str,
    protocol_sha256: str,
    windows_manifest: Path,
) -> tuple[dict[str, Any], str]:
    report_path = audit_root / "snapshot_reuse_audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("protocol_id") not in {None, protocol_id}:
        raise ValueError("Resumed audit protocol ID differs from the batch.")
    if report.get("protocol_sha256") not in {None, protocol_sha256}:
        raise ValueError("Resumed audit protocol checksum differs from the batch.")
    if report.get("windows_manifest_sha256") not in {
        None,
        sha256_file(windows_manifest),
    }:
        raise ValueError("Resumed audit Windows manifest checksum drifted.")
    companions = (
        (
            "repeatability_comparisons_path",
            "repeatability_comparisons_sha256",
            "repeatability_comparisons.jsonl",
        ),
        (
            "verification_dataset_metrics_path",
            "verification_dataset_metrics_sha256",
            "verification_dataset_metrics.jsonl",
        ),
    )
    for path_field, checksum_field, filename in companions:
        companion = audit_root / filename
        expected = report.get(checksum_field)
        if expected is not None:
            if not companion.is_file() or sha256_file(companion) != expected:
                raise ValueError(f"Resumed audit companion drifted: {filename}")
            report[path_field] = filename
    return report, _atomic_json(report_path, report)


def audit_snapshot_batch(
    *,
    batch_directory: Path,
    windows_results_root: Path,
    output_directory: Path,
    auditor: Callable[..., dict[str, Any]] = audit_snapshot_reuse_execution,
) -> dict[str, Any]:
    batch_directory = batch_directory.resolve()
    windows_results_root = windows_results_root.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    temporary.mkdir(parents=True, exist_ok=True)
    try:
        jobs_path = batch_directory / "jobs.jsonl"
        batch_manifest_path = batch_directory / "batch_manifest.json"
        batch_manifest = json.loads(
            batch_manifest_path.read_text(encoding="utf-8-sig")
        )
        if sha256_file(jobs_path) != batch_manifest["jobs_sha256"]:
            raise ValueError("Windows batch jobs checksum drifted.")
        jobs = _read_jsonl(jobs_path)
        if len(jobs) != int(batch_manifest["job_count"]):
            raise ValueError("Windows batch job count does not match its manifest.")
        expected_worker_sha256 = sha256_file(
            batch_directory / "run_licensed_snapshot_reuse.ps1"
        )

        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for job in jobs:
            key = str(job["protocol_key"])
            protocol_path = batch_directory / str(job["protocol_relative_path"])
            result_root = windows_results_root / key
            audit_root = temporary / "audits" / key
            record: dict[str, Any] = {
                "protocol_key": key,
                "protocol_id": str(job["protocol_id"]),
                "protocol_sha256": str(job["protocol_sha256"]),
                "full_name": str(job["full_name"]),
                "version_id": job.get("version_id"),
                "repository_commit": job.get("repository_commit"),
                "compounds": job.get("compounds") or [],
                "protocol_relative_path": str(job["protocol_relative_path"]),
                "windows_result_relative_path": key,
                "audit_relative_path": f"audits/{key}/snapshot_reuse_audit.json",
            }
            try:
                if sha256_file(protocol_path) != str(job["protocol_sha256"]):
                    raise ValueError("Protocol checksum differs from jobs.jsonl.")
                run_manifest = result_root / "windows_run_manifest.json"
                if not run_manifest.is_file():
                    raise FileNotFoundError(
                        f"Windows result manifest is missing: {run_manifest}"
                    )
                run_manifest_value = json.loads(
                    run_manifest.read_text(encoding="utf-8-sig")
                )
                if (
                    run_manifest_value.get("worker_script_sha256")
                    != expected_worker_sha256
                ):
                    raise ValueError(
                        "Windows result worker checksum differs from the locked batch worker."
                    )
                report_path = audit_root / "snapshot_reuse_audit.json"
                audit: dict[str, Any]
                if report_path.is_file():
                    try:
                        report, report_sha256 = _canonicalize_audit_report(
                            audit_root=audit_root,
                            protocol_id=str(job["protocol_id"]),
                            protocol_sha256=str(job["protocol_sha256"]),
                            windows_manifest=run_manifest,
                        )
                        audit = {**report, "report_sha256": report_sha256}
                        record["resumed"] = True
                    except (FileNotFoundError, ValueError, json.JSONDecodeError):
                        shutil.rmtree(audit_root, ignore_errors=True)
                        audit = auditor(
                            protocol_path=protocol_path,
                            windows_result_root=result_root,
                            output_directory=audit_root,
                        )
                else:
                    if audit_root.exists():
                        shutil.rmtree(audit_root, ignore_errors=True)
                    audit = auditor(
                        protocol_path=protocol_path,
                        windows_result_root=result_root,
                        output_directory=audit_root,
                    )
                report, report_sha256 = _canonicalize_audit_report(
                    audit_root=audit_root,
                    protocol_id=str(job["protocol_id"]),
                    protocol_sha256=str(job["protocol_sha256"]),
                    windows_manifest=run_manifest,
                )
                audit = {**audit, **report, "report_sha256": report_sha256}
                engineering_accepted = bool(audit.get("engineering_accepted"))
                scientific_accepted = bool(audit.get("scientific_accepted"))
                if scientific_accepted:
                    audit_status = "audited_passed"
                else:
                    audit_status = "audited_negative"
                record.update(
                    {
                        "audit_status": audit_status,
                        "engineering_accepted": engineering_accepted,
                        "scientific_accepted": scientific_accepted,
                        "validation_grade": audit.get("validation_grade"),
                        "run_label_grade": int(
                            audit.get("run_label_grade") or 0
                        ),
                        "ranking_label": audit.get("ranking_label"),
                        "reusable_candidate": bool(
                            audit.get("reusable_candidate")
                        ),
                        "audit_report_sha256": str(audit["report_sha256"]),
                        "windows_manifest_sha256": str(
                            audit["windows_manifest_sha256"]
                        ),
                        "evaluated_dataset_count": int(
                            (audit.get("counts") or {}).get(
                                "evaluated_datasets", 0
                            )
                        ),
                        "dataset_error_count": int(
                            (audit.get("counts") or {}).get("dataset_errors", 0)
                        ),
                    }
                )
            except (FileNotFoundError, ValueError) as exc:
                record.update(
                    {
                        "audit_status": "runner_failed",
                        "engineering_accepted": False,
                        "scientific_accepted": False,
                        "validation_grade": None,
                        "run_label_grade": None,
                        "ranking_label": None,
                        "reusable_candidate": False,
                        "failure_type": type(exc).__name__,
                        "failure_detail": str(exc),
                    }
                )
                failures.append(record.copy())
            except Exception as exc:
                record.update(
                    {
                        "audit_status": "auditor_failed",
                        "engineering_accepted": False,
                        "scientific_accepted": False,
                        "validation_grade": None,
                        "run_label_grade": None,
                        "ranking_label": None,
                        "reusable_candidate": False,
                        "failure_type": type(exc).__name__,
                        "failure_detail": str(exc),
                    }
                )
                failures.append(record.copy())
            rows.append(record)

        rows.sort(key=lambda row: str(row["protocol_key"]))
        failures.sort(key=lambda row: str(row["protocol_key"]))
        index_sha = _atomic_jsonl(temporary / "audit_index.jsonl", rows)
        failures_sha = _atomic_jsonl(temporary / "failures.jsonl", failures)
        counts = {
            "job_count": len(rows),
            "audited_count": sum(
                str(row["audit_status"]).startswith("audited_") for row in rows
            ),
            "scientific_passed_count": sum(
                row["audit_status"] == "audited_passed" for row in rows
            ),
            "executed_negative_count": sum(
                row["audit_status"] == "audited_negative" for row in rows
            ),
            "runner_failed_count": sum(
                row["audit_status"] == "runner_failed" for row in rows
            ),
            "auditor_failed_count": sum(
                row["audit_status"] == "auditor_failed" for row in rows
            ),
        }
        summary = {
            "audit_batch_version": AUDIT_BATCH_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "batch_directory": str(batch_directory),
            "batch_manifest_sha256": sha256_file(batch_manifest_path),
            "jobs_sha256": sha256_file(jobs_path),
            "windows_results_root": str(windows_results_root),
            "counts": counts,
            "output_checksums": {
                "audit_index.jsonl": index_sha,
                "failures.jsonl": failures_sha,
            },
            "complete": (
                counts["runner_failed_count"] == 0
                and counts["auditor_failed_count"] == 0
            ),
        }
        (temporary / "orchestrator_failure.json").unlink(missing_ok=True)
        _atomic_json(temporary / "summary.json", summary)
        os.replace(temporary, output_directory)
        return summary
    except Exception as exc:
        try:
            _atomic_json(
                temporary / "orchestrator_failure.json",
                {
                    "audit_batch_version": AUDIT_BATCH_VERSION,
                    "failed_at": datetime.now(UTC).isoformat(),
                    "failure_type": type(exc).__name__,
                    "failure_detail": str(exc),
                },
            )
        except Exception:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit every checksum-locked Windows snapshot execution."
    )
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--windows-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_snapshot_batch(
        batch_directory=args.batch,
        windows_results_root=args.windows_results,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
