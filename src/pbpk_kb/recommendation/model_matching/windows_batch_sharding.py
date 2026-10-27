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
from pbpk_kb.recommendation.model_matching.windows_snapshot_batch import (
    build_windows_snapshot_batch,
)


SHARD_BUILDER_VERSION = "pbpk-windows-batch-sharding-1.0.0"
SHARD_MERGER_VERSION = "pbpk-windows-batch-shard-merger-1.0.0"


class WindowsBatchShardingError(RuntimeError):
    pass


TERMINAL_BATCH_STATUSES = {
    "executed_passed": "passed",
    "already_executed_passed": "passed",
    "executed_negative": "failed",
    "already_executed_negative": "failed",
}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise WindowsBatchShardingError(f"Required JSON file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise WindowsBatchShardingError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise WindowsBatchShardingError(f"Required JSONL file is missing: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise WindowsBatchShardingError(f"Expected JSON objects in: {path}")
    return rows


def _unique_job_map(
    rows: list[dict[str, Any]], *, source: Path
) -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("protocol_key") or "")
        if not key or key in jobs:
            raise WindowsBatchShardingError(
                f"Protocol keys must be non-empty and unique in {source}."
            )
        jobs[key] = row
    return jobs


def _tree_inventory(root: Path) -> tuple[list[dict[str, Any]], str, int]:
    files: list[dict[str, Any]] = []
    byte_size = 0
    for path in sorted(
        (value for value in root.rglob("*") if value.is_file()),
        key=lambda value: value.relative_to(root).as_posix().casefold(),
    ):
        size = path.stat().st_size
        byte_size += size
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "byte_size": size,
                "sha256": sha256_file(path),
            }
        )
    return files, hashlib.sha256(_canonical_bytes(files)).hexdigest(), byte_size


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def deterministic_shard(protocol_key: str, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    digest = hashlib.sha256(protocol_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def partition_protocol_rows(
    rows: list[dict[str, Any]], shard_count: int
) -> list[list[dict[str, Any]]]:
    if not rows:
        raise WindowsBatchShardingError("No protocol rows were supplied.")
    keys = [str(row.get("protocol_key") or "") for row in rows]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise WindowsBatchShardingError(
            "Protocol keys must be non-empty and globally unique."
        )
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for row in rows:
        key = str(row["protocol_key"])
        shards[deterministic_shard(key, shard_count)].append(row)
    for shard in shards:
        shard.sort(key=lambda row: str(row["protocol_key"]))
    if any(not shard for shard in shards):
        raise WindowsBatchShardingError(
            "Every configured shard must contain at least one protocol."
        )
    assigned = [str(row["protocol_key"]) for shard in shards for row in shard]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(keys):
        raise WindowsBatchShardingError(
            "Shard inventories are not mutually exclusive and complete."
        )
    return shards


def build_windows_snapshot_shards(
    *,
    project_root: Path,
    protocols_path: Path,
    runtime_archive: Path,
    output_directory: Path,
    shard_count: int = 4,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    rows = [
        json.loads(line)
        for line in protocols_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    shards = partition_protocol_rows(rows, shard_count)
    output_directory.mkdir(parents=True)
    try:
        shard_records: list[dict[str, Any]] = []
        for index, shard_rows in enumerate(shards):
            shard_root = output_directory / f"shard_{index:02d}"
            shard_root.mkdir()
            shard_protocols = shard_root / "protocols.jsonl"
            shard_protocols_sha256 = _atomic_jsonl(
                shard_protocols, shard_rows
            )
            batch_root = shard_root / "bundle"
            batch = build_windows_snapshot_batch(
                project_root=project_root,
                protocols_path=shard_protocols,
                runtime_archive=runtime_archive,
                output_directory=batch_root,
            )
            keys = [str(row["protocol_key"]) for row in shard_rows]
            shard_records.append(
                {
                    "shard_index": index,
                    "protocol_count": len(keys),
                    "protocol_keys": keys,
                    "protocol_inventory_sha256": hashlib.sha256(
                        _canonical_bytes(keys)
                    ).hexdigest(),
                    "protocols_sha256": shard_protocols_sha256,
                    "batch_relative_path": batch_root.relative_to(
                        output_directory
                    ).as_posix(),
                    "batch_manifest_sha256": sha256_file(
                        batch_root / "batch_manifest.json"
                    ),
                    "jobs_sha256": batch["jobs_sha256"],
                    "runtime_sha256": batch["runtime_sha256"],
                }
            )
        all_keys = sorted(str(row["protocol_key"]) for row in rows)
        manifest = {
            "shard_builder_version": SHARD_BUILDER_VERSION,
            "assignment_rule": "uint64_be(SHA256(protocol_key)[0:8]) mod shard_count",
            "shard_count": shard_count,
            "protocol_count": len(all_keys),
            "protocol_inventory_sha256": hashlib.sha256(
                _canonical_bytes(all_keys)
            ).hexdigest(),
            "source_protocols_path": str(protocols_path.resolve()),
            "source_protocols_sha256": sha256_file(protocols_path),
            "runtime_sha256": sha256_file(runtime_archive),
            "shards": shard_records,
            "inventory_mutually_exclusive": True,
            "inventory_complete": True,
            "created_at": datetime.now(UTC).isoformat(),
        }
        _atomic_json(output_directory / "shard_manifest.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise


def merge_windows_snapshot_shard_results(
    *,
    full_batch_directory: Path,
    shard_directory: Path,
    shard_results_directory: Path,
    output_directory: Path,
    materialization: str = "hardlink",
) -> dict[str, Any]:
    """Validate and merge terminal Windows shard results into one audit input."""

    full_batch_directory = full_batch_directory.resolve()
    shard_directory = shard_directory.resolve()
    shard_results_directory = shard_results_directory.resolve()
    output_directory = output_directory.resolve()
    if materialization not in {"hardlink", "symlink"}:
        raise ValueError("materialization must be 'hardlink' or 'symlink'.")
    if output_directory.exists():
        raise FileExistsError(output_directory)
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        raise WindowsBatchShardingError(
            f"Stale temporary merge directory exists: {temporary}"
        )

    full_manifest_path = full_batch_directory / "batch_manifest.json"
    full_jobs_path = full_batch_directory / "jobs.jsonl"
    full_manifest = _read_json(full_manifest_path)
    if sha256_file(full_jobs_path) != str(full_manifest.get("jobs_sha256") or ""):
        raise WindowsBatchShardingError("Full Windows batch jobs checksum drifted.")
    full_jobs = _unique_job_map(_read_jsonl(full_jobs_path), source=full_jobs_path)
    if len(full_jobs) != int(full_manifest.get("job_count") or -1):
        raise WindowsBatchShardingError("Full Windows batch job count drifted.")

    shard_manifest_path = shard_directory / "shard_manifest.json"
    shard_manifest = _read_json(shard_manifest_path)
    shard_records = list(shard_manifest.get("shards") or [])
    shard_count = int(shard_manifest.get("shard_count") or 0)
    if shard_count <= 0 or len(shard_records) != shard_count:
        raise WindowsBatchShardingError("Shard manifest count is inconsistent.")
    if int(shard_manifest.get("protocol_count") or -1) != len(full_jobs):
        raise WindowsBatchShardingError(
            "Shard manifest and full batch protocol counts differ."
        )

    temporary.mkdir(parents=True)
    merged_records: list[dict[str, Any]] = []
    assigned_keys: set[str] = set()
    hardlinked_file_count = 0
    copied_file_count = 0
    symlinked_protocol_count = 0
    try:
        for record in sorted(
            shard_records, key=lambda value: int(value["shard_index"])
        ):
            shard_index = int(record["shard_index"])
            batch_root = shard_directory / str(record["batch_relative_path"])
            batch_manifest_path = batch_root / "batch_manifest.json"
            if sha256_file(batch_manifest_path) != str(
                record["batch_manifest_sha256"]
            ):
                raise WindowsBatchShardingError(
                    f"Shard {shard_index} batch manifest checksum drifted."
                )
            batch_manifest = _read_json(batch_manifest_path)
            jobs_path = batch_root / "jobs.jsonl"
            jobs_sha256 = sha256_file(jobs_path)
            if jobs_sha256 != str(record["jobs_sha256"]) or jobs_sha256 != str(
                batch_manifest.get("jobs_sha256") or ""
            ):
                raise WindowsBatchShardingError(
                    f"Shard {shard_index} jobs checksum drifted."
                )
            jobs = _unique_job_map(_read_jsonl(jobs_path), source=jobs_path)
            expected_keys = set(str(key) for key in record["protocol_keys"])
            if set(jobs) != expected_keys or len(jobs) != int(
                record["protocol_count"]
            ):
                raise WindowsBatchShardingError(
                    f"Shard {shard_index} job inventory drifted."
                )
            if any(
                deterministic_shard(key, shard_count) != shard_index
                for key in expected_keys
            ):
                raise WindowsBatchShardingError(
                    f"Shard {shard_index} violates the deterministic assignment rule."
                )
            if assigned_keys.intersection(expected_keys):
                raise WindowsBatchShardingError("Shard protocol inventories overlap.")
            assigned_keys.update(expected_keys)

            results_root = shard_results_directory / f"results_{shard_index:02d}"
            summary_path = results_root / "batch_summary.json"
            progress_path = results_root / "batch_progress.json"
            summary = _read_json(summary_path)
            progress_value = json.loads(progress_path.read_text(encoding="utf-8-sig"))
            if not isinstance(progress_value, list):
                progress_value = [progress_value]
            summary_jobs = _unique_job_map(
                list(summary.get("jobs") or []), source=summary_path
            )
            progress_jobs = _unique_job_map(progress_value, source=progress_path)
            if set(summary_jobs) != expected_keys or set(progress_jobs) != expected_keys:
                raise WindowsBatchShardingError(
                    f"Shard {shard_index} result inventory is incomplete."
                )
            if int(summary.get("job_count") or -1) != len(expected_keys):
                raise WindowsBatchShardingError(
                    f"Shard {shard_index} summary job count drifted."
                )
            expected_worker_sha256 = sha256_file(
                batch_root / "run_licensed_snapshot_reuse.ps1"
            )
            for key in sorted(expected_keys):
                status = str(summary_jobs[key].get("status") or "")
                if status not in TERMINAL_BATCH_STATUSES:
                    raise WindowsBatchShardingError(
                        f"Shard {shard_index} protocol {key} is not terminal: {status}"
                    )
                if str(progress_jobs[key].get("status") or "") != status:
                    raise WindowsBatchShardingError(
                        f"Shard {shard_index} progress and summary disagree for {key}."
                    )
                source_result = results_root / key
                run_manifest_path = source_result / "windows_run_manifest.json"
                run_manifest = _read_json(run_manifest_path)
                job = jobs[key]
                if (
                    str(run_manifest.get("protocol_id") or "")
                    != str(job["protocol_id"])
                    or str(run_manifest.get("protocol_sha256") or "")
                    != str(job["protocol_sha256"])
                    or str(run_manifest.get("worker_script_sha256") or "")
                    != expected_worker_sha256
                    or str(run_manifest.get("status") or "")
                    != TERMINAL_BATCH_STATUSES[status]
                ):
                    raise WindowsBatchShardingError(
                        f"Shard {shard_index} result identity mismatch for {key}."
                    )
                destination = temporary / key
                source_files, source_inventory_sha256, source_byte_size = (
                    _tree_inventory(source_result)
                )
                if materialization == "symlink":
                    destination.symlink_to(source_result, target_is_directory=True)
                    if destination.resolve() != source_result.resolve():
                        raise WindowsBatchShardingError(
                            f"Merged result symlink drift for {key}."
                        )
                    files = source_files
                    inventory_sha256 = source_inventory_sha256
                    byte_size = source_byte_size
                    symlinked_protocol_count += 1
                else:
                    shutil.copytree(
                        source_result,
                        destination,
                        copy_function=_hardlink_or_copy,
                    )
                    files, inventory_sha256, byte_size = _tree_inventory(destination)
                    if (
                        files != source_files
                        or inventory_sha256 != source_inventory_sha256
                        or byte_size != source_byte_size
                    ):
                        raise WindowsBatchShardingError(
                            f"Merged result checksum drift for {key}."
                        )
                    for source_file in source_result.rglob("*"):
                        if not source_file.is_file():
                            continue
                        destination_file = destination / source_file.relative_to(
                            source_result
                        )
                        if source_file.stat().st_ino == destination_file.stat().st_ino:
                            hardlinked_file_count += 1
                        else:
                            copied_file_count += 1
                merged_records.append(
                    {
                        "protocol_key": key,
                        "protocol_id": str(job["protocol_id"]),
                        "protocol_sha256": str(job["protocol_sha256"]),
                        "source_shard_index": shard_index,
                        "terminal_status": status,
                        "windows_status": str(run_manifest["status"]),
                        "windows_manifest_sha256": sha256_file(run_manifest_path),
                        "file_count": len(files),
                        "byte_size": byte_size,
                        "tree_inventory_sha256": inventory_sha256,
                    }
                )

        if assigned_keys != set(full_jobs):
            missing = sorted(set(full_jobs) - assigned_keys)
            unexpected = sorted(assigned_keys - set(full_jobs))
            raise WindowsBatchShardingError(
                f"Merged inventory differs from the full batch; missing={missing}, "
                f"unexpected={unexpected}."
            )
        if len(merged_records) != len(full_jobs):
            raise WindowsBatchShardingError("Merged protocol count drifted.")

        merged_records.sort(key=lambda value: str(value["protocol_key"]))
        records_sha256 = _atomic_jsonl(
            temporary / "merged_results.jsonl", merged_records
        )
        counts = {
            "protocol_count": len(merged_records),
            "passed_count": sum(
                row["windows_status"] == "passed" for row in merged_records
            ),
            "negative_count": sum(
                row["windows_status"] == "failed" for row in merged_records
            ),
            "hardlinked_file_count": hardlinked_file_count,
            "copied_file_count": copied_file_count,
            "symlinked_protocol_count": symlinked_protocol_count,
            "byte_size": sum(int(row["byte_size"]) for row in merged_records),
        }
        merge_manifest = {
            "shard_merger_version": SHARD_MERGER_VERSION,
            "full_batch_directory": str(full_batch_directory),
            "full_batch_manifest_sha256": sha256_file(full_manifest_path),
            "full_batch_jobs_sha256": sha256_file(full_jobs_path),
            "shard_directory": str(shard_directory),
            "shard_manifest_sha256": sha256_file(shard_manifest_path),
            "shard_results_directory": str(shard_results_directory),
            "merged_results_sha256": records_sha256,
            "inventory_mutually_exclusive": True,
            "inventory_complete": True,
            "all_results_terminal": True,
            "materialization": materialization,
            "counts": counts,
            "created_at": datetime.now(UTC).isoformat(),
        }
        _atomic_json(temporary / "merge_manifest.json", merge_manifest)
        os.replace(temporary, output_directory)
        return merge_manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or merge deterministic Windows execution shards."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    build_parser.add_argument("--protocols", type=Path, required=True)
    build_parser.add_argument("--runtime-archive", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--shard-count", type=int, default=4)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--full-batch", type=Path, required=True)
    merge_parser.add_argument("--shards", type=Path, required=True)
    merge_parser.add_argument("--shard-results", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument(
        "--materialization", choices=("hardlink", "symlink"), default="hardlink"
    )
    args = parser.parse_args()
    if args.command == "build":
        result = build_windows_snapshot_shards(
            project_root=args.project_root,
            protocols_path=args.protocols,
            runtime_archive=args.runtime_archive,
            output_directory=args.output,
            shard_count=args.shard_count,
        )
    else:
        result = merge_windows_snapshot_shard_results(
            full_batch_directory=args.full_batch,
            shard_directory=args.shards,
            shard_results_directory=args.shard_results,
            output_directory=args.output,
            materialization=args.materialization,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
