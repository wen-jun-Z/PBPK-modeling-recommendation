from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from pbpk_kb.model_assets.license_policy import sha256_file


BATCH_BUILDER_VERSION = "pbpk-windows-snapshot-batch-1.2.0"


class WindowsSnapshotBatchError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (
                json.dumps(row, ensure_ascii=True, sort_keys=True, default=str) + "\n"
            ).encode("utf-8")
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _copy_locked(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file():
        raise WindowsSnapshotBatchError(f"Required source file is missing: {source}")
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise WindowsSnapshotBatchError(
            f"Checksum mismatch for {source}: expected {expected_sha256}, found {actual}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected_sha256:
        raise WindowsSnapshotBatchError(f"Copied file checksum drift: {destination}")


def build_windows_snapshot_batch(
    *,
    project_root: Path,
    protocols_path: Path,
    runtime_archive: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    temporary = output_directory.with_name(output_directory.name + ".tmp")
    if temporary.exists():
        raise WindowsSnapshotBatchError(f"Stale temporary directory: {temporary}")
    temporary.mkdir(parents=True)
    try:
        rows = _read_jsonl(protocols_path)
        if not rows:
            raise WindowsSnapshotBatchError("No planned protocols were supplied.")
        runner_source = project_root / "scripts/windows/run_licensed_snapshot_reuse.ps1"
        batch_source = project_root / "scripts/windows/run_snapshot_protocol_batch.ps1"
        shutil.copy2(runner_source, temporary / runner_source.name)
        shutil.copy2(batch_source, temporary / batch_source.name)
        jobs: list[dict[str, Any]] = []
        runtime_hashes: set[str] = set()
        for row in rows:
            key = str(row["protocol_key"])
            protocol_source = Path(str(row["protocol_path"]))
            protocol = json.loads(protocol_source.read_text(encoding="utf-8"))
            protocol_sha = str(row["protocol_sha256"])
            runtime_hashes.add(str(protocol["runtime"]["pksim_portable_sha256"]))
            checkout = Path(str(row["checkout_path"]))
            snapshot_source = checkout / str(row["snapshot_relative_path"])
            snapshot_sha = str(protocol["source"]["snapshot_sha256"])
            protocol_relative = Path("protocols") / key / "snapshot_reuse_protocol.json"
            snapshot_relative = Path("snapshots") / key / snapshot_source.name
            _copy_locked(protocol_source, temporary / protocol_relative, protocol_sha)
            _copy_locked(snapshot_source, temporary / snapshot_relative, snapshot_sha)
            jobs.append(
                {
                    "protocol_key": key,
                    "protocol_id": protocol["protocol_id"],
                    "full_name": row["full_name"],
                    "version_id": row.get("version_id"),
                    "repository_commit": row["repository_commit"],
                    "model_name": row["model_name"],
                    "compounds": row.get("compounds") or [],
                    "protocol_relative_path": protocol_relative.as_posix(),
                    "protocol_sha256": protocol_sha,
                    "snapshot_relative_path": snapshot_relative.as_posix(),
                    "snapshot_sha256": snapshot_sha,
                    "simulation_count": len(protocol["simulations"]),
                    "verification_simulation_count": row["model_verification_count"],
                    "quantitative_verification_series_count": sum(
                        (1 if simulation.get("evaluation_series") else 0)
                        + len(simulation.get("additional_evaluation_series") or [])
                        for simulation in protocol["simulations"]
                    ),
                }
            )
        if len(runtime_hashes) != 1:
            raise WindowsSnapshotBatchError(
                f"Protocols require different runtimes: {sorted(runtime_hashes)}"
            )
        expected_runtime_sha = next(iter(runtime_hashes))
        runtime_relative = Path("runtime") / runtime_archive.name
        _copy_locked(runtime_archive, temporary / runtime_relative, expected_runtime_sha)
        jobs.sort(
            key=lambda row: (
                str(row["full_name"]).casefold(),
                str(row["repository_commit"]),
            )
        )
        jobs_sha = _atomic_jsonl(temporary / "jobs.jsonl", jobs)
        files = []
        for path in sorted(
            (value for value in temporary.rglob("*") if value.is_file()),
            key=lambda value: value.relative_to(temporary).as_posix().casefold(),
        ):
            files.append(
                {
                    "relative_path": path.relative_to(temporary).as_posix(),
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "batch_builder_version": BATCH_BUILDER_VERSION,
            "protocols_path": str(protocols_path.resolve()),
            "protocols_sha256": sha256_file(protocols_path),
            "job_count": len(jobs),
            "simulation_count": sum(int(row["simulation_count"]) for row in jobs),
            "verification_simulation_count": sum(
                int(row["verification_simulation_count"]) for row in jobs
            ),
            "quantitative_verification_series_count": sum(
                int(row["quantitative_verification_series_count"]) for row in jobs
            ),
            "runtime_relative_path": runtime_relative.as_posix(),
            "runtime_sha256": expected_runtime_sha,
            "jobs_sha256": jobs_sha,
            "files": files,
            "created_at": datetime.now(UTC).isoformat(),
        }
        _atomic_json(temporary / "batch_manifest.json", manifest)
        os.replace(temporary, output_directory)
        archive_path = output_directory.with_suffix(".zip")
        if archive_path.exists():
            raise FileExistsError(archive_path)
        with zipfile.ZipFile(
            archive_path,
            "x",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for path in sorted(
                (value for value in output_directory.rglob("*") if value.is_file()),
                key=lambda value: value.relative_to(output_directory).as_posix().casefold(),
            ):
                archive.write(
                    path,
                    arcname=(
                        f"{output_directory.name}/"
                        f"{path.relative_to(output_directory).as_posix()}"
                    ),
                )
        manifest["archive_path"] = str(archive_path)
        manifest["archive_sha256"] = sha256_file(archive_path)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a checksum-locked Windows batch for snapshot validation protocols."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocols", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_windows_snapshot_batch(
        project_root=args.project_root,
        protocols_path=args.protocols,
        runtime_archive=args.runtime_archive,
        output_directory=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
