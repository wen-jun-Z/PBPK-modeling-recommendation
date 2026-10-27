from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any


class ResultBundleError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
    ):
        raise ResultBundleError(f"Unsafe result-relative path: {value!r}")
    return path


def _result_files(result: dict[str, Any]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for project in result.get("projects", []):
        artifacts = [
            *project.get("artifacts", []),
            project.get("stdout_artifact"),
            project.get("stderr_artifact"),
        ]
        for simulation in project.get("simulation_results", []):
            artifacts.extend(
                [
                    simulation.get("stdout_artifact"),
                    simulation.get("stderr_artifact"),
                ]
            )
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ResultBundleError("Execution result has an invalid artifact.")
            files.append(artifact)
    return files


def _zip_info(name: str, *, size: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.file_size = size
    return info


def build_result_bundle(
    *,
    result_path: Path,
    output_archive: Path,
) -> dict[str, Any]:
    result_path = result_path.resolve()
    result_root = result_path.parent
    if result_path.name != "execution_result.json":
        raise ResultBundleError("Result file must be named execution_result.json.")
    sidecar_path = result_root / "execution_result.sha256"
    if not sidecar_path.is_file():
        raise ResultBundleError("Execution result checksum sidecar is missing.")
    result_sha256 = _sha256_file(result_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sidecar.get("file") != result_path.name
        or sidecar.get("sha256") != result_sha256
    ):
        raise ResultBundleError("Execution result checksum sidecar does not match.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    job_id = result.get("job_id")
    if not isinstance(job_id, str) or not job_id.startswith("osp-exec-"):
        raise ResultBundleError("Execution result job ID is invalid.")
    if result.get("status") != "passed":
        raise ResultBundleError("Only a passed execution result can be bundled.")

    root_name = job_id
    artifact_records = _result_files(result)
    files: list[dict[str, Any]] = [
        {
            "relative_path": result_path.name,
            "byte_size": result_path.stat().st_size,
            "sha256": result_sha256,
            "kind": "execution_result",
        },
        {
            "relative_path": sidecar_path.name,
            "byte_size": sidecar_path.stat().st_size,
            "sha256": _sha256_file(sidecar_path),
            "kind": "execution_result_checksum",
        },
    ]
    seen = {result_path.name, sidecar_path.name}
    for artifact in artifact_records:
        relative = _safe_relative_path(str(artifact.get("relative_path", "")))
        normalized = relative.as_posix()
        if normalized in seen:
            raise ResultBundleError(f"Duplicate result artifact: {normalized}")
        seen.add(normalized)
        path = (result_root / Path(*relative.parts)).resolve()
        try:
            path.relative_to(result_root)
        except ValueError as exc:
            raise ResultBundleError(
                f"Result artifact escapes result root: {normalized}"
            ) from exc
        if not path.is_file():
            raise ResultBundleError(f"Result artifact is missing: {normalized}")
        expected_size = int(artifact.get("byte_size", -1))
        expected_sha256 = str(artifact.get("sha256", ""))
        if path.stat().st_size != expected_size:
            raise ResultBundleError(f"Result artifact size mismatch: {normalized}")
        if _sha256_file(path) != expected_sha256:
            raise ResultBundleError(
                f"Result artifact checksum mismatch: {normalized}"
            )
        files.append(
            {
                "relative_path": normalized,
                "byte_size": expected_size,
                "sha256": expected_sha256,
                "kind": "execution_artifact",
            }
        )

    files.sort(key=lambda item: str(item["relative_path"]).casefold())
    manifest = {
        "bundle_manifest_schema_version": "1.0.0",
        "job_id": job_id,
        "root_name": root_name,
        "execution_result_sha256": result_sha256,
        "file_count": len(files),
        "total_uncompressed_bytes": sum(
            int(item["byte_size"]) for item in files
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "files": files,
    }
    manifest_bytes = _canonical_json(manifest)
    output_archive = output_archive.resolve()
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    if output_archive.exists():
        raise ResultBundleError(f"Output archive already exists: {output_archive}")
    temporary = output_archive.with_suffix(output_archive.suffix + ".tmp")
    if temporary.exists():
        raise ResultBundleError(f"Temporary archive already exists: {temporary}")

    try:
        with zipfile.ZipFile(
            temporary,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            manifest_name = f"{root_name}/bundle_manifest.json"
            archive.writestr(
                _zip_info(manifest_name, size=len(manifest_bytes)),
                manifest_bytes,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
            for index, record in enumerate(files, start=1):
                relative = _safe_relative_path(str(record["relative_path"]))
                source = result_root / Path(*relative.parts)
                member_name = f"{root_name}/{relative.as_posix()}"
                digest = hashlib.sha256()
                copied = 0
                info = _zip_info(member_name, size=int(record["byte_size"]))
                with source.open("rb") as input_handle, archive.open(
                    info,
                    mode="w",
                    force_zip64=True,
                ) as output_handle:
                    while chunk := input_handle.read(4 * 1024 * 1024):
                        output_handle.write(chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                if (
                    copied != int(record["byte_size"])
                    or digest.hexdigest() != record["sha256"]
                ):
                    raise ResultBundleError(
                        f"Source changed during bundling: {relative.as_posix()}"
                    )
                if index % 100 == 0 or index == len(files):
                    print(
                        json.dumps(
                            {
                                "phase": "bundle",
                                "files_complete": index,
                                "files_total": len(files),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

        with zipfile.ZipFile(temporary, mode="r") as archive:
            expected_names = {
                f"{root_name}/bundle_manifest.json",
                *{
                    f"{root_name}/{record['relative_path']}"
                    for record in files
                },
            }
            actual_names = {info.filename for info in archive.infolist()}
            if actual_names != expected_names:
                raise ResultBundleError("Bundle member coverage mismatch.")
            for index, record in enumerate(files, start=1):
                member_name = f"{root_name}/{record['relative_path']}"
                digest = hashlib.sha256()
                copied = 0
                with archive.open(member_name, "r") as handle:
                    while chunk := handle.read(4 * 1024 * 1024):
                        digest.update(chunk)
                        copied += len(chunk)
                if (
                    copied != int(record["byte_size"])
                    or digest.hexdigest() != record["sha256"]
                ):
                    raise ResultBundleError(
                        f"Bundle content verification failed: {member_name}"
                    )
                if index % 100 == 0 or index == len(files):
                    print(
                        json.dumps(
                            {
                                "phase": "verify",
                                "files_complete": index,
                                "files_total": len(files),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        temporary.replace(output_archive)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    archive_sha256 = _sha256_file(output_archive)
    sidecar_payload = {
        "bundle_sidecar_schema_version": "1.0.0",
        "job_id": job_id,
        "file": output_archive.name,
        "byte_size": output_archive.stat().st_size,
        "sha256": archive_sha256,
        "root_name": root_name,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    bundle_sidecar = output_archive.with_suffix(output_archive.suffix + ".sha256.json")
    sidecar_temporary = bundle_sidecar.with_suffix(bundle_sidecar.suffix + ".tmp")
    with sidecar_temporary.open("xb") as handle:
        handle.write(_canonical_json(sidecar_payload))
        handle.flush()
        os.fsync(handle.fileno())
    sidecar_temporary.replace(bundle_sidecar)
    return {
        **sidecar_payload,
        "archive_path": str(output_archive),
        "sidecar_path": str(bundle_sidecar),
        "file_count": len(files),
        "total_uncompressed_bytes": manifest["total_uncompressed_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create and verify a checksum-locked ZIP64 bundle from a passed "
            "Windows OSP execution result."
        )
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_result_bundle(
        result_path=args.result,
        output_archive=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
