from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from pbpk_kb.model_assets.license_policy import sha256_file


RECONCILER_VERSION = "pbpk-perturbation-protocol-reconciler-1.0.0"


class PerturbationReconciliationError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PerturbationReconciliationError(f"Required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise PerturbationReconciliationError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PerturbationReconciliationError(f"Required JSONL is missing: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise PerturbationReconciliationError(f"Expected JSON objects: {path}")
    return rows


def _row_map(rows: list[dict[str, Any]], *, source: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("protocol_key") or "")
        if not key or key in values:
            raise PerturbationReconciliationError(
                f"Protocol keys must be non-empty and unique in {source}."
            )
        values[key] = row
    return values


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


def protocol_execution_semantic_sha256(protocol: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(protocol))
    value.pop("generated_at", None)
    value.pop("protocol_builder_version", None)
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validated_plan_rows(directory: Path) -> dict[str, dict[str, Any]]:
    manifest = _read_json(directory / "manifest.json")
    path = directory / "protocols.jsonl"
    expected = str((manifest.get("output_checksums") or {}).get("protocols.jsonl") or "")
    if sha256_file(path) != expected:
        raise PerturbationReconciliationError(
            f"Perturbation protocol inventory checksum drifted: {path}"
        )
    return _row_map(_read_jsonl(path), source=path)


def reconcile_perturbation_protocols(
    *,
    project_root: Path,
    previous_plan_directory: Path,
    corrected_plan_directory: Path,
    previous_batch_directory: Path,
    previous_results_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    previous_plan_directory = previous_plan_directory.resolve()
    corrected_plan_directory = corrected_plan_directory.resolve()
    previous_batch_directory = previous_batch_directory.resolve()
    previous_results_root = previous_results_root.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(output_directory)
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        raise PerturbationReconciliationError(
            f"Stale reconciliation directory exists: {temporary}"
        )

    previous_rows = _validated_plan_rows(previous_plan_directory)
    corrected_rows = _validated_plan_rows(corrected_plan_directory)
    if set(previous_rows) != set(corrected_rows):
        raise PerturbationReconciliationError(
            "Previous and corrected protocol inventories differ."
        )

    batch_manifest_path = previous_batch_directory / "batch_manifest.json"
    batch_manifest = _read_json(batch_manifest_path)
    batch_jobs_path = previous_batch_directory / "jobs.jsonl"
    if sha256_file(batch_jobs_path) != str(batch_manifest.get("jobs_sha256") or ""):
        raise PerturbationReconciliationError("Previous batch jobs checksum drifted.")
    previous_jobs = _row_map(_read_jsonl(batch_jobs_path), source=batch_jobs_path)
    if set(previous_jobs) != set(previous_rows):
        raise PerturbationReconciliationError(
            "Previous plan and executed batch inventories differ."
        )
    current_worker_sha256 = sha256_file(
        project_root / "scripts/windows/run_licensed_snapshot_reuse.ps1"
    )
    previous_worker_sha256 = sha256_file(
        previous_batch_directory / "run_licensed_snapshot_reuse.ps1"
    )
    worker_is_reusable = current_worker_sha256 == previous_worker_sha256

    temporary.mkdir(parents=True)
    try:
        reconciled: list[dict[str, Any]] = []
        reused: list[dict[str, Any]] = []
        rerun: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for key in sorted(previous_rows):
            previous_row = previous_rows[key]
            corrected_row = corrected_rows[key]
            previous_protocol_path = Path(str(previous_row["protocol_path"]))
            corrected_protocol_path = Path(str(corrected_row["protocol_path"]))
            if sha256_file(previous_protocol_path) != str(
                previous_row["protocol_sha256"]
            ):
                raise PerturbationReconciliationError(
                    f"Previous protocol checksum drifted: {key}"
                )
            if sha256_file(corrected_protocol_path) != str(
                corrected_row["protocol_sha256"]
            ):
                raise PerturbationReconciliationError(
                    f"Corrected protocol checksum drifted: {key}"
                )
            previous_protocol = _read_json(previous_protocol_path)
            corrected_protocol = _read_json(corrected_protocol_path)
            previous_fingerprint = protocol_execution_semantic_sha256(
                previous_protocol
            )
            corrected_fingerprint = protocol_execution_semantic_sha256(
                corrected_protocol
            )
            execution_semantics_unchanged = (
                previous_fingerprint == corrected_fingerprint
            )
            reusable = execution_semantics_unchanged and worker_is_reusable
            reason_code = "EXECUTION_SEMANTICS_UNCHANGED"
            if not execution_semantics_unchanged:
                reason_code = "EXECUTION_SEMANTICS_CHANGED"
            elif not worker_is_reusable:
                reason_code = "WORKER_CHECKSUM_CHANGED"

            if reusable:
                previous_job = previous_jobs[key]
                if (
                    str(previous_job["protocol_id"])
                    != str(previous_row["protocol_id"])
                    or str(previous_job["protocol_sha256"])
                    != str(previous_row["protocol_sha256"])
                ):
                    raise PerturbationReconciliationError(
                        f"Previous batch identity drifted for {key}."
                    )
                run_manifest_path = (
                    previous_results_root / key / "windows_run_manifest.json"
                )
                run_manifest = _read_json(run_manifest_path)
                if (
                    str(run_manifest.get("protocol_id") or "")
                    != str(previous_row["protocol_id"])
                    or str(run_manifest.get("protocol_sha256") or "")
                    != str(previous_row["protocol_sha256"])
                    or str(run_manifest.get("worker_script_sha256") or "")
                    != previous_worker_sha256
                    or str(run_manifest.get("status") or "") != "passed"
                ):
                    raise PerturbationReconciliationError(
                        f"Previous Windows result is not identity-locked and reusable: {key}"
                    )
                selected = previous_row
                reused.append(selected)
            else:
                selected = corrected_row
                rerun.append(selected)
            reconciled.append(selected)
            decisions.append(
                {
                    "protocol_key": key,
                    "decision": "reuse_previous" if reusable else "rerun_corrected",
                    "reason_code": reason_code,
                    "previous_protocol_id": str(previous_row["protocol_id"]),
                    "previous_protocol_sha256": str(
                        previous_row["protocol_sha256"]
                    ),
                    "previous_execution_semantic_sha256": previous_fingerprint,
                    "corrected_protocol_id": str(corrected_row["protocol_id"]),
                    "corrected_protocol_sha256": str(
                        corrected_row["protocol_sha256"]
                    ),
                    "corrected_execution_semantic_sha256": corrected_fingerprint,
                    "selected_protocol_id": str(selected["protocol_id"]),
                    "selected_protocol_sha256": str(selected["protocol_sha256"]),
                    "scientific_grade_generated": False,
                }
            )

        outputs = {
            "protocols.jsonl": _atomic_jsonl(
                temporary / "protocols.jsonl", reconciled
            ),
            "reused_protocols.jsonl": _atomic_jsonl(
                temporary / "reused_protocols.jsonl", reused
            ),
            "rerun_protocols.jsonl": _atomic_jsonl(
                temporary / "rerun_protocols.jsonl", rerun
            ),
            "reconciliation.jsonl": _atomic_jsonl(
                temporary / "reconciliation.jsonl", decisions
            ),
        }
        manifest = {
            "reconciler_version": RECONCILER_VERSION,
            "previous_plan_directory": str(previous_plan_directory),
            "corrected_plan_directory": str(corrected_plan_directory),
            "previous_batch_directory": str(previous_batch_directory),
            "previous_results_root": str(previous_results_root),
            "current_worker_sha256": current_worker_sha256,
            "previous_worker_sha256": previous_worker_sha256,
            "counts": {
                "protocol_count": len(reconciled),
                "reuse_previous_count": len(reused),
                "rerun_corrected_count": len(rerun),
            },
            "inventory_complete": len(reconciled) == len(previous_rows),
            "scientific_grade_generated": False,
            "output_checksums": outputs,
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
        description="Reconcile corrected perturbation protocols with reusable results."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--previous-plan", type=Path, required=True)
    parser.add_argument("--corrected-plan", type=Path, required=True)
    parser.add_argument("--previous-batch", type=Path, required=True)
    parser.add_argument("--previous-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile_perturbation_protocols(
        project_root=args.project_root,
        previous_plan_directory=args.previous_plan,
        corrected_plan_directory=args.corrected_plan,
        previous_batch_directory=args.previous_batch,
        previous_results_root=args.previous_results,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
