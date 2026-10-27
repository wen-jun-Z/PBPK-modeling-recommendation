from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from pbpk_kb.model_assets.license_policy import sha256_file


RESULT_RECONCILER_VERSION = "pbpk-perturbation-result-reconciler-1.0.0"


class PerturbationResultReconciliationError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PerturbationResultReconciliationError(
            f"Required JSON file is missing: {path}"
        )
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise PerturbationResultReconciliationError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PerturbationResultReconciliationError(
            f"Required JSONL file is missing: {path}"
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _row_map(rows: list[dict[str, Any]], *, source: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("protocol_key") or "")
        if not key or key in values:
            raise PerturbationResultReconciliationError(
                f"Protocol keys must be non-empty and unique in {source}."
            )
        values[key] = row
    return values


def _validated_batch_jobs(directory: Path) -> dict[str, dict[str, Any]]:
    manifest = _read_json(directory / "batch_manifest.json")
    jobs_path = directory / "jobs.jsonl"
    if sha256_file(jobs_path) != str(manifest.get("jobs_sha256") or ""):
        raise PerturbationResultReconciliationError(
            f"Batch jobs checksum drifted: {directory}"
        )
    jobs = _row_map(_read_jsonl(jobs_path), source=jobs_path)
    if len(jobs) != int(manifest.get("job_count") or -1):
        raise PerturbationResultReconciliationError(
            f"Batch job count drifted: {directory}"
        )
    return jobs


def _atomic_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
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


def assemble_reconciled_perturbation_results(
    *,
    reconciliation_directory: Path,
    full_batch_directory: Path,
    previous_results_root: Path,
    correction_batch_directory: Path,
    correction_results_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    reconciliation_directory = reconciliation_directory.resolve()
    full_batch_directory = full_batch_directory.resolve()
    previous_results_root = previous_results_root.resolve()
    correction_batch_directory = correction_batch_directory.resolve()
    correction_results_root = correction_results_root.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        raise PerturbationResultReconciliationError(
            f"Stale result reconciliation directory exists: {temporary}"
        )

    reconciliation_manifest = _read_json(
        reconciliation_directory / "manifest.json"
    )
    decisions_path = reconciliation_directory / "reconciliation.jsonl"
    expected_decisions_sha = str(
        (reconciliation_manifest.get("output_checksums") or {}).get(
            "reconciliation.jsonl"
        )
        or ""
    )
    if sha256_file(decisions_path) != expected_decisions_sha:
        raise PerturbationResultReconciliationError(
            "Protocol reconciliation decisions checksum drifted."
        )
    decisions = _row_map(_read_jsonl(decisions_path), source=decisions_path)
    full_jobs = _validated_batch_jobs(full_batch_directory)
    correction_jobs = _validated_batch_jobs(correction_batch_directory)
    if set(decisions) != set(full_jobs):
        raise PerturbationResultReconciliationError(
            "Reconciliation decisions and full batch inventories differ."
        )
    rerun_keys = {
        key
        for key, row in decisions.items()
        if row.get("decision") == "rerun_corrected"
    }
    reuse_keys = {
        key
        for key, row in decisions.items()
        if row.get("decision") == "reuse_previous"
    }
    if rerun_keys | reuse_keys != set(full_jobs) or rerun_keys & reuse_keys:
        raise PerturbationResultReconciliationError(
            "Reconciliation decisions are not complete and mutually exclusive."
        )
    if set(correction_jobs) != rerun_keys:
        raise PerturbationResultReconciliationError(
            "Correction batch does not exactly cover rerun decisions."
        )

    full_worker_sha = sha256_file(
        full_batch_directory / "run_licensed_snapshot_reuse.ps1"
    )
    correction_worker_sha = sha256_file(
        correction_batch_directory / "run_licensed_snapshot_reuse.ps1"
    )
    if full_worker_sha != correction_worker_sha:
        raise PerturbationResultReconciliationError(
            "Full and correction batch worker checksums differ."
        )

    temporary.mkdir(parents=True)
    try:
        records: list[dict[str, Any]] = []
        for key in sorted(full_jobs):
            job = full_jobs[key]
            decision = decisions[key]
            if (
                str(job["protocol_id"])
                != str(decision["selected_protocol_id"])
                or str(job["protocol_sha256"])
                != str(decision["selected_protocol_sha256"])
            ):
                raise PerturbationResultReconciliationError(
                    f"Full batch selection identity drifted for {key}."
                )
            source_role = (
                "correction" if key in rerun_keys else "previous_execution"
            )
            source_root = (
                correction_results_root if key in rerun_keys else previous_results_root
            )
            source = source_root / key
            run_manifest_path = source / "windows_run_manifest.json"
            run_manifest = _read_json(run_manifest_path)
            if (
                str(run_manifest.get("protocol_id") or "")
                != str(job["protocol_id"])
                or str(run_manifest.get("protocol_sha256") or "")
                != str(job["protocol_sha256"])
                or str(run_manifest.get("worker_script_sha256") or "")
                != full_worker_sha
                or str(run_manifest.get("status") or "") != "passed"
            ):
                raise PerturbationResultReconciliationError(
                    f"Selected Windows result identity mismatch for {key}."
                )
            destination = temporary / key
            destination.symlink_to(source, target_is_directory=True)
            if destination.resolve() != source.resolve():
                raise PerturbationResultReconciliationError(
                    f"Result symlink target drifted for {key}."
                )
            records.append(
                {
                    "protocol_key": key,
                    "protocol_id": str(job["protocol_id"]),
                    "protocol_sha256": str(job["protocol_sha256"]),
                    "source_role": source_role,
                    "source_result_root": str(source),
                    "windows_manifest_sha256": sha256_file(run_manifest_path),
                    "windows_status": "passed",
                }
            )
        records_sha = _atomic_jsonl(
            temporary / "reconciled_results.jsonl", records
        )
        manifest = {
            "result_reconciler_version": RESULT_RECONCILER_VERSION,
            "reconciliation_directory": str(reconciliation_directory),
            "full_batch_directory": str(full_batch_directory),
            "previous_results_root": str(previous_results_root),
            "correction_batch_directory": str(correction_batch_directory),
            "correction_results_root": str(correction_results_root),
            "worker_sha256": full_worker_sha,
            "counts": {
                "protocol_count": len(records),
                "reused_previous_count": len(reuse_keys),
                "corrected_count": len(rerun_keys),
                "passed_count": len(records),
            },
            "inventory_complete": len(records) == len(full_jobs),
            "inventory_mutually_exclusive": True,
            "all_results_terminal": True,
            "reconciled_results_sha256": records_sha,
            "created_at": datetime.now(UTC).isoformat(),
        }
        _atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output_directory)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble previous and corrected perturbation results."
    )
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--full-batch", type=Path, required=True)
    parser.add_argument("--previous-results", type=Path, required=True)
    parser.add_argument("--correction-batch", type=Path, required=True)
    parser.add_argument("--correction-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble_reconciled_perturbation_results(
        reconciliation_directory=args.reconciliation,
        full_batch_directory=args.full_batch,
        previous_results_root=args.previous_results,
        correction_batch_directory=args.correction_batch,
        correction_results_root=args.correction_results,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
