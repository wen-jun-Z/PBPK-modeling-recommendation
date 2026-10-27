from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from pydantic import Field

from pbpk_kb.model_assets.license_policy import (
    resolve_repository_license,
    sha256_file,
)
from pbpk_kb.model_assets.models import (
    AssetKind,
    EvaluationReportSummary,
    LicenseDecision,
    ModelAsset,
    ParseStatus,
    RepositoryInventorySummary,
    RepositorySnapshot,
    StrictModel,
)
from pbpk_kb.model_assets.osp_project_parser import OSPProjectError, parse_osp_project
from pbpk_kb.model_assets.report_parser import (
    parse_markdown_report,
    parse_pdf_report,
)
from pbpk_kb.model_assets.repository_adapter import inspect_repository


INVENTORY_VERSION = "1.2.0"
SOURCE_ID = "osp_pbpk_model_library"
REPOSITORY_URL = "https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library"


class PinnedSnapshotAsset(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_location: str = Field(min_length=1)
    derived_from_relative_path: str | None = None
    derived_from_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class PinnedSnapshotManifest(StrictModel):
    manifest_schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    source_id: str = Field(min_length=1)
    repository_url: str = Field(min_length=1)
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    assets: list[PinnedSnapshotAsset] = Field(min_length=1)


def _safe_snapshot_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe pinned snapshot path: {value!r}")
    return path.as_posix()


def _asset_kind(path: Path) -> AssetKind:
    suffix = path.suffix.casefold()
    name = path.name.casefold()
    if name in {".gitignore", ".gitattributes", ".gitmodules"}:
        return AssetKind.REPOSITORY_METADATA
    if suffix == ".pksim5":
        return AssetKind.PKSIM_PROJECT
    if suffix == ".pkml":
        return AssetKind.PKML_SIMULATION
    if suffix == ".pdf" and ("evaluation" in name or "report" in name):
        return AssetKind.EVALUATION_REPORT_PDF
    if suffix == ".md" and ("evaluation" in name or "report" in name):
        return AssetKind.EVALUATION_REPORT_MARKDOWN
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".ico"}:
        return AssetKind.IMAGE
    if suffix in {".csv", ".xlsx", ".xls", ".json"}:
        return AssetKind.DATASET
    if name.startswith("license") or name.startswith("copying"):
        return AssetKind.LICENSE
    if suffix in {".py", ".r", ".yml", ".yaml"}:
        return AssetKind.WORKFLOW
    if suffix in {".md", ".txt", ".html"}:
        return AssetKind.DOCUMENTATION
    return AssetKind.OTHER


def _media_type(path: Path, kind: AssetKind) -> str:
    if kind == AssetKind.PKSIM_PROJECT:
        return "application/vnd.open-systems-pharmacology.pksim-project+sqlite3"
    if kind == AssetKind.PKML_SIMULATION:
        return "application/vnd.open-systems-pharmacology.pkml+xml"
    if kind == AssetKind.REPOSITORY_METADATA:
        return "text/plain"
    guessed, _encoding = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _asset_id(source_id: str, commit: str, relative_path: str) -> str:
    value = f"{source_id}\n{commit}\n{relative_path}".encode()
    return hashlib.sha256(value).hexdigest()


def _model_family(relative_path: str) -> str | None:
    parts = Path(relative_path).parts
    return parts[0] if len(parts) > 1 else None


def _json_line(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, records: Iterable[Any]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            encoded = (_json_line(record) + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def build_repository_inventory(
    *,
    repository_root: Path,
    output_directory: Path,
    source_id: str = SOURCE_ID,
    repository_url: str = REPOSITORY_URL,
    repository_commit: str | None = None,
    default_branch: str = "main",
    snapshot_manifest_path: Path | None = None,
) -> RepositoryInventorySummary:
    repository_root = repository_root.resolve()
    pinned_manifest: PinnedSnapshotManifest | None = None
    if repository_commit is None:
        if snapshot_manifest_path is not None:
            raise ValueError(
                "A snapshot manifest requires an explicit repository commit."
            )
        snapshot = inspect_repository(
            repository_root,
            source_id=source_id,
            repository_url=repository_url,
        )
        snapshot_mode = "git_checkout"
    else:
        if snapshot_manifest_path is None:
            raise ValueError(
                "Pinned file snapshots require a checksum manifest."
            )
        pinned_manifest = PinnedSnapshotManifest.model_validate_json(
            snapshot_manifest_path.read_text(encoding="utf-8")
        )
        if (
            pinned_manifest.source_id != source_id
            or pinned_manifest.repository_url != repository_url
            or pinned_manifest.repository_commit != repository_commit
        ):
            raise ValueError(
                "Pinned snapshot identity does not match inventory arguments."
            )
        snapshot = RepositorySnapshot(
            source_id=source_id,
            repository_url=repository_url,
            repository_path=str(repository_root),
            commit_sha=repository_commit,
            default_branch=default_branch,
            is_clean=True,
            captured_at=datetime.now(UTC),
        )
        snapshot_mode = "pinned_files"
    repository_license = resolve_repository_license(repository_root)
    assets: list[ModelAsset] = []
    project_results = []
    report_results: list[EvaluationReportSummary] = []
    failures: list[dict[str, Any]] = []

    tracked_paths = sorted(
        (
            path
            for path in repository_root.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(repository_root).parts
        ),
        key=lambda path: path.relative_to(repository_root).as_posix().casefold(),
    )
    source_locations: dict[str, str] = {}
    if pinned_manifest is not None:
        manifest_assets: dict[str, PinnedSnapshotAsset] = {}
        for asset in pinned_manifest.assets:
            relative_path = _safe_snapshot_path(asset.relative_path)
            if relative_path in manifest_assets:
                raise ValueError(
                    f"Duplicate pinned snapshot path: {relative_path}"
                )
            manifest_assets[relative_path] = asset
        tracked_relative_paths = {
            path.relative_to(repository_root).as_posix()
            for path in tracked_paths
        }
        if tracked_relative_paths != set(manifest_assets):
            raise ValueError(
                "Pinned snapshot manifest and filesystem path sets differ."
            )
        for relative_path, asset in manifest_assets.items():
            if sha256_file(repository_root / relative_path) != asset.sha256:
                raise ValueError(
                    f"Pinned snapshot checksum drift: {relative_path}"
                )
            if (asset.derived_from_relative_path is None) != (
                asset.derived_from_sha256 is None
            ):
                raise ValueError(
                    "Pinned snapshot derivation requires both parent path "
                    "and checksum."
                )
            if asset.derived_from_relative_path is not None:
                parent_path = _safe_snapshot_path(
                    asset.derived_from_relative_path
                )
                parent = manifest_assets.get(parent_path)
                if (
                    parent is None
                    or parent.sha256 != asset.derived_from_sha256
                ):
                    raise ValueError(
                        "Pinned snapshot derivation parent mismatch: "
                        f"{relative_path}"
                    )
            source_locations[relative_path] = asset.source_location

    output_directory.mkdir(parents=True, exist_ok=False)
    for path in tracked_paths:
        relative_path = path.relative_to(repository_root).as_posix()
        kind = _asset_kind(path)
        parse_status = ParseStatus.UNSUPPORTED
        parse_error: str | None = None
        try:
            if kind == AssetKind.PKSIM_PROJECT:
                parsed = parse_osp_project(
                    path,
                    repository_commit=snapshot.commit_sha,
                )
                project_results.append(parsed)
                parse_status = parsed.parse_status
            elif kind == AssetKind.EVALUATION_REPORT_MARKDOWN:
                report_results.append(
                    parse_markdown_report(
                        path,
                        relative_path=relative_path,
                        repository_commit=snapshot.commit_sha,
                    )
                )
                parse_status = ParseStatus.PARSED
            elif kind == AssetKind.EVALUATION_REPORT_PDF:
                report_results.append(parse_pdf_report(path, relative_path=relative_path))
                parse_status = ParseStatus.PARSED
            elif kind in {
                AssetKind.IMAGE,
                AssetKind.DATASET,
                AssetKind.WORKFLOW,
                AssetKind.DOCUMENTATION,
                AssetKind.LICENSE,
                AssetKind.REPOSITORY_METADATA,
            }:
                parse_status = ParseStatus.PARSED
        except Exception as exc:
            parse_status = ParseStatus.FAILED
            parse_error = f"{type(exc).__name__}: {exc}"
            failures.append(
                {
                    "relative_path": relative_path,
                    "asset_kind": kind.value,
                    "stage": "parse",
                    "error": parse_error,
                }
            )
        assets.append(
            ModelAsset(
                asset_id=_asset_id(source_id, snapshot.commit_sha, relative_path),
                model_family=_model_family(relative_path),
                relative_path=relative_path,
                asset_kind=kind,
                byte_size=path.stat().st_size,
                sha256=sha256_file(path),
                media_type=_media_type(path, kind),
                source_id=source_id,
                source_location=source_locations.get(
                    relative_path,
                    f"GITHUB_BLOB:{snapshot.commit_sha}/{relative_path}",
                ),
                repository_commit=snapshot.commit_sha,
                license=LicenseDecision.model_validate(repository_license.model_dump()),
                parse_status=parse_status,
                parse_error=parse_error,
            )
        )

    output_checksums: dict[str, str] = {}
    output_checksums["assets.jsonl"] = _write_jsonl(
        output_directory / "assets.jsonl", assets
    )
    output_checksums["projects.jsonl"] = _write_jsonl(
        output_directory / "projects.jsonl", project_results
    )
    output_checksums["reports.jsonl"] = _write_jsonl(
        output_directory / "reports.jsonl", report_results
    )
    output_checksums["failures.jsonl"] = _write_jsonl(
        output_directory / "failures.jsonl", failures
    )

    kind_counts = Counter(asset.asset_kind.value for asset in assets)
    status_counts = Counter(asset.parse_status.value for asset in assets)
    schema_counts = Counter(result.schema_fingerprint for result in project_results)
    project_count = kind_counts[AssetKind.PKSIM_PROJECT.value]
    blocking_reasons: list[str] = []
    if repository_license.status.value == "unresolved":
        blocking_reasons.append(repository_license.reason_code)
    failed_projects = sum(
        1
        for asset in assets
        if asset.asset_kind == AssetKind.PKSIM_PROJECT
        and asset.parse_status == ParseStatus.FAILED
    )
    if failed_projects:
        blocking_reasons.append("PROJECT_PARSE_FAILURES")
    blocking_reasons.append("OSP_WINDOWS_EXECUTION_PROBE_NOT_RUN")
    gate1_static_acceptance = (
        project_count > 0
        and len(project_results) == project_count
        and failed_projects == 0
        and all(asset.sha256 for asset in assets)
    )
    summary = RepositoryInventorySummary(
        inventory_version=INVENTORY_VERSION,
        source_id=source_id,
        repository_commit=snapshot.commit_sha,
        generated_at=datetime.now(UTC),
        repository_snapshot=snapshot,
        repository_license=repository_license,
        asset_count=len(assets),
        model_project_count=project_count,
        parsed_project_count=len(project_results),
        failed_project_count=failed_projects,
        report_count=len(report_results),
        total_bytes=sum(asset.byte_size for asset in assets),
        asset_kind_counts=dict(sorted(kind_counts.items())),
        parse_status_counts=dict(sorted(status_counts.items())),
        project_schema_fingerprints=dict(sorted(schema_counts.items())),
        output_checksums=output_checksums,
        gate1_static_acceptance=gate1_static_acceptance,
        execution_probe_complete=False,
        blocking_reasons=blocking_reasons,
        metadata={
            "repository_clean": snapshot.is_clean,
            "repository_snapshot_mode": snapshot_mode,
            "snapshot_manifest_sha256": (
                sha256_file(snapshot_manifest_path)
                if snapshot_manifest_path is not None
                else None
            ),
            "asset_paths_are_relative": True,
            "outputs_are_immutable": True,
        },
    )
    summary_path = output_directory / "summary.json"
    _write_json(summary_path, summary)
    return summary
