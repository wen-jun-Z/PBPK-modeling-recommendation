from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from pbpk_kb.parsers.emitter import PARSER_VERSION, ParseEmitter, stable_id
from pbpk_kb.parsers.file_types import OOXML_CANONICAL_EXTENSIONS, extension_for, file_family_for, ooxml_package_family
from pbpk_kb.parsers.models import ParseAsset, ParseOutcome
from pbpk_kb.parsers.raw_formats import RawFormatParsers
from pbpk_kb.parsers.registry import ParserRegistry
from pbpk_kb.processing.raw_processing_paths import RawProcessingLayout, raw_processing_layout


MAX_ARCHIVE_MEMBERS = 50_000
MAX_ARCHIVE_MEMBER_BYTES = 1_073_741_824
MAX_ARCHIVE_TOTAL_BYTES = 10_737_418_240


def prepare_asset_for_dispatch(asset: ParseAsset) -> tuple[ParseAsset, list[str], dict[str, Any]]:
    if asset.physical_path.is_file():
        with asset.physical_path.open("rb") as stream:
            header = stream.read(512)
        logical_name = PurePosixPath(asset.inner_path or asset.storage_path).name
        kind = None
        if (logical_name.startswith("._") and header[:8] == bytes.fromhex("0005160700020000")):
            kind = "appledouble_resource_metadata"
        elif (logical_name.startswith("~$") and asset.file_size <= 4096 and b"\x00" in header
                and Path(logical_name).suffix.lower() in {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}
                and not header.startswith((b"PK", bytes.fromhex("d0cf11e0a1b11ae1")))
                and not header.lstrip().startswith((b"<", b"{\\rtf"))):
            kind = "office_owner_lock"
        if kind:
            metadata = {"non_content_file_kind": kind, "declared_file_family": asset.file_family}
            return replace(asset, file_family="filesystem_metadata",
                           asset_metadata_json={**asset.asset_metadata_json, **metadata}), [kind], metadata
        # Legacy Word files may contain an embedded ZIP holding only theme data.
        # Read the outer signature before attempting any OOXML ZIP dispatch.
        if asset.file_extension.lower() == ".docx" and header.startswith(bytes.fromhex("d0cf11e0a1b11ae1")):
            metadata = {"detected_format": "legacy_word_ole", "declared_file_extension": asset.file_extension}
            return replace(asset, file_family="document", file_extension=".doc",
                           asset_metadata_json={**asset.asset_metadata_json, **metadata}), ["Legacy Word content with .docx suffix"], metadata
        if (asset.file_family in {"document", "table", "presentation"}
                and header.lstrip().lower().startswith((b"<!doctype html", b"<html"))):
            metadata = {"detected_format": "html", "declared_file_extension": asset.file_extension}
            return replace(asset, file_family="html", file_extension=".html",
                           asset_metadata_json={**asset.asset_metadata_json, **metadata}), ["HTML content with Office suffix"], metadata
    detected_family = ooxml_package_family(asset.physical_path)
    if detected_family is None or detected_family == asset.file_family:
        return asset, [], {}
    warning = (
        "OOXML package type differs from declared file family: "
        f"declared={asset.file_family} detected={detected_family} extension={asset.file_extension}"
    )
    metadata = {
        "declared_file_family": asset.file_family,
        "detected_file_family": detected_family,
        "declared_file_extension": asset.file_extension,
        "ooxml_type_mismatch": True,
    }
    return (
        replace(
            asset,
            file_family=detected_family,
            file_extension=OOXML_CANONICAL_EXTENSIONS[detected_family],
            asset_metadata_json={**asset.asset_metadata_json, **metadata},
        ),
        [warning],
        metadata,
    )


class ArchiveParser:
    def __init__(
        self,
        registry: ParserRegistry,
        standalone_checksums: dict[str, str],
        staging_root: Path,
    ) -> None:
        self.registry = registry
        self.standalone_checksums = standalone_checksums
        self.staging_root = staging_root
        self.seen_member_checksums: dict[str, str] = {}

    def parse(self, asset: ParseAsset, emitter: ParseEmitter) -> ParseOutcome:
        depth = int(asset.asset_metadata_json.get("archive_depth") or 0)
        if depth >= 2:
            return ParseOutcome(status="unsupported", warnings=["Nested archive depth limit reached"])
        path = asset.physical_path
        if zipfile.is_zipfile(path):
            return self._parse_zip(asset, emitter, depth)
        if tarfile.is_tarfile(path):
            return self._parse_tar(asset, emitter, depth)
        if is_rar_archive(path):
            return self._parse_libarchive(asset, emitter, depth)
        return ParseOutcome(status="unsupported", warnings=[f"No safe extractor available for {asset.file_extension}"])

    def _parse_zip(self, asset: ParseAsset, emitter: ParseEmitter, depth: int) -> ParseOutcome:
        member_count = 0
        failed_count = 0
        unsupported_count = 0
        expanded_bytes = 0
        with zipfile.ZipFile(asset.physical_path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ARCHIVE_MEMBERS:
                raise ValueError(f"Archive member count exceeds limit: {len(infos)}")
            for info in infos:
                if info.is_dir():
                    continue
                member_count += 1
                unsafe_reason = archive_path_rejection(info.filename)
                mode = info.external_attr >> 16
                if mode and stat.S_ISLNK(mode):
                    unsafe_reason = "symbolic_link"
                expanded_bytes += int(info.file_size)
                if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    unsafe_reason = "member_size_limit"
                if expanded_bytes > MAX_ARCHIVE_TOTAL_BYTES:
                    unsafe_reason = "archive_expanded_size_limit"
                if unsafe_reason:
                    self._write_rejected_member(asset, emitter, info.filename, info.compress_size, info.file_size, unsafe_reason)
                    failed_count += 1
                    continue
                with archive.open(info, "r") as stream:
                    status = self._parse_member(
                        asset,
                        emitter,
                        stream,
                        inner_path=info.filename,
                        compressed_size=info.compress_size,
                        expected_size=info.file_size,
                        depth=depth,
                    )
                if status == "failed":
                    failed_count += 1
                elif status == "unsupported":
                    unsupported_count += 1
        status = "failed" if failed_count else "parsed" if member_count else "metadata_only"
        return ParseOutcome(
            status=status,
            warnings=[f"unsupported_members={unsupported_count}"] if unsupported_count else [],
            metadata={"member_count": member_count, "failed_member_count": failed_count, "unsupported_member_count": unsupported_count},
        )

    def _parse_tar(self, asset: ParseAsset, emitter: ParseEmitter, depth: int) -> ParseOutcome:
        member_count = 0
        failed_count = 0
        unsupported_count = 0
        expanded_bytes = 0
        with tarfile.open(asset.physical_path, "r:*") as archive:
            members = archive.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ValueError(f"Archive member count exceeds limit: {len(members)}")
            for info in members:
                if not info.isfile():
                    if info.issym() or info.islnk() or info.isdev():
                        self._write_rejected_member(asset, emitter, info.name, None, info.size, "non_regular_member")
                        failed_count += 1
                    continue
                member_count += 1
                unsafe_reason = archive_path_rejection(info.name)
                expanded_bytes += int(info.size)
                if info.size > MAX_ARCHIVE_MEMBER_BYTES:
                    unsafe_reason = "member_size_limit"
                if expanded_bytes > MAX_ARCHIVE_TOTAL_BYTES:
                    unsafe_reason = "archive_expanded_size_limit"
                if unsafe_reason:
                    self._write_rejected_member(asset, emitter, info.name, None, info.size, unsafe_reason)
                    failed_count += 1
                    continue
                stream = archive.extractfile(info)
                if stream is None:
                    self._write_rejected_member(asset, emitter, info.name, None, info.size, "member_stream_unavailable")
                    failed_count += 1
                    continue
                with stream:
                    status = self._parse_member(
                        asset,
                        emitter,
                        stream,
                        inner_path=info.name,
                        compressed_size=None,
                        expected_size=info.size,
                        depth=depth,
                    )
                if status == "failed":
                    failed_count += 1
                elif status == "unsupported":
                    unsupported_count += 1
        status = "failed" if failed_count else "parsed" if member_count else "metadata_only"
        return ParseOutcome(
            status=status,
            warnings=[f"unsupported_members={unsupported_count}"] if unsupported_count else [],
            metadata={"member_count": member_count, "failed_member_count": failed_count, "unsupported_member_count": unsupported_count},
        )

    def _parse_libarchive(self, asset: ParseAsset, emitter: ParseEmitter, depth: int) -> ParseOutcome:
        tool = shutil.which("bsdtar")
        if tool is None:
            return ParseOutcome(status="unsupported", warnings=["RAR archive requires the read-only bsdtar/libarchive reader"])
        environment = {**os.environ, "LC_ALL": "C"}
        listing = subprocess.run(
            [tool, "-tvf", str(asset.physical_path)],
            capture_output=True,
            check=False,
            timeout=300,
            env=environment,
        )
        if listing.returncode != 0:
            error = listing.stderr.decode("utf-8", errors="replace")[:2000]
            raise ValueError(f"bsdtar could not list archive: {error}")
        members = parse_bsdtar_verbose_listing(listing.stdout.decode("utf-8", errors="replace"))
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"Archive member count exceeds limit: {len(members)}")

        member_count = 0
        failed_count = 0
        unsupported_count = 0
        expanded_bytes = 0
        self.staging_root.mkdir(parents=True, exist_ok=True)
        for member in members:
            mode = str(member["mode"])
            inner_path = str(member["inner_path"])
            expected_size = int(member["size"])
            if mode.startswith("d"):
                continue
            member_count += 1
            unsafe_reason = archive_path_rejection(inner_path)
            if not mode.startswith("-"):
                unsafe_reason = "non_regular_member"
            expanded_bytes += expected_size
            if expected_size > MAX_ARCHIVE_MEMBER_BYTES:
                unsafe_reason = "member_size_limit"
            if expanded_bytes > MAX_ARCHIVE_TOTAL_BYTES:
                unsafe_reason = "archive_expanded_size_limit"
            if unsafe_reason:
                self._write_rejected_member(asset, emitter, inner_path, None, expected_size, unsafe_reason)
                failed_count += 1
                continue

            suffix = Path(inner_path).suffix[:20]
            extracted_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.staging_root, suffix=suffix, delete=False) as extracted:
                    extracted_path = Path(extracted.name)
                    completed = subprocess.run(
                        [tool, "-xOf", str(asset.physical_path), "--", inner_path],
                        stdout=extracted,
                        stderr=subprocess.PIPE,
                        check=False,
                        timeout=900,
                        env=environment,
                    )
                actual_size = extracted_path.stat().st_size
                if completed.returncode != 0 or actual_size != expected_size:
                    detail = completed.stderr.decode("utf-8", errors="replace")[:1000]
                    reason = (
                        f"libarchive_extract_failed returncode={completed.returncode} "
                        f"expected_size={expected_size} actual_size={actual_size} detail={detail}"
                    )
                    self._write_rejected_member(
                        asset,
                        emitter,
                        inner_path,
                        None,
                        actual_size,
                        reason,
                        failure_type="archive_member_extract_failed",
                    )
                    failed_count += 1
                    continue
                with extracted_path.open("rb") as stream:
                    status = self._parse_member(
                        asset,
                        emitter,
                        stream,
                        inner_path=inner_path,
                        compressed_size=None,
                        expected_size=expected_size,
                        depth=depth,
                    )
                if status == "failed":
                    failed_count += 1
                elif status == "unsupported":
                    unsupported_count += 1
            finally:
                if extracted_path is not None:
                    extracted_path.unlink(missing_ok=True)

        status = "failed" if failed_count else "parsed" if member_count else "metadata_only"
        warnings = [f"unsupported_members={unsupported_count}"] if unsupported_count else []
        return ParseOutcome(
            status=status,
            warnings=warnings,
            metadata={
                "archive_reader": "bsdtar/libarchive",
                "member_count": member_count,
                "failed_member_count": failed_count,
                "unsupported_member_count": unsupported_count,
            },
        )

    def _parse_member(
        self,
        parent: ParseAsset,
        emitter: ParseEmitter,
        stream: BinaryIO,
        *,
        inner_path: str,
        compressed_size: int | None,
        expected_size: int,
        depth: int,
    ) -> str:
        suffix = Path(inner_path).suffix[:20]
        self.staging_root.mkdir(parents=True, exist_ok=True)
        digest = sha256()
        byte_count = 0
        temporary_path: Path | None = None
        try:
            logical_inner_path = f"{parent.inner_path}!{inner_path}" if parent.inner_path else inner_path
            with tempfile.NamedTemporaryFile(dir=self.staging_root, suffix=suffix, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    byte_count += len(chunk)
                    if byte_count > MAX_ARCHIVE_MEMBER_BYTES:
                        raise ValueError("member_size_limit")
                    digest.update(chunk)
                    temporary.write(chunk)
            member_checksum = digest.hexdigest()
            extension = extension_for(Path(inner_path), None, None)
            family = file_family_for(temporary_path, extension, None, parent.source_id)
            member_id = stable_id("archive_member", parent.raw_asset_id, logical_inner_path, member_checksum)
            duplicate_target = self.standalone_checksums.get(member_checksum)
            duplicate_member = self.seen_member_checksums.get(member_checksum)
            status = "duplicate_linked" if duplicate_target or duplicate_member else "parsed"
            parser_name: str | None = None
            warnings: list[str] = []
            dispatch_metadata: dict[str, Any] = {}
            if status != "duplicate_linked":
                child = ParseAsset(
                    **{
                        **parent.__dict__,
                        "read_path": str(temporary_path),
                        "inner_path": logical_inner_path,
                        "file_extension": extension,
                        "file_family": family,
                        "file_size": byte_count,
                        "asset_metadata_json": {
                            **parent.asset_metadata_json,
                            "archive_depth": depth + 1,
                            "archive_member_checksum": member_checksum,
                        },
                    }
                )
                dispatch_child, dispatch_warnings, dispatch_metadata = prepare_asset_for_dispatch(child)
                try:
                    registered = self.registry.resolve(dispatch_child)
                    parser_name = registered.name
                    outcome = registered.callback(dispatch_child, emitter)
                    status = outcome.status
                    warnings = [*dispatch_warnings, *outcome.warnings]
                except Exception as exc:
                    status = "failed"
                    warnings = [*dispatch_warnings, str(exc)]
                    emitter.failure(
                        parent,
                        parser_name=parser_name or "archive_member_dispatch",
                        failure_type="archive_member_parse_failed",
                        error=str(exc),
                        source_location=emitter.location(parent, f"inner_path={logical_inner_path}"),
                        payload={"inner_path": logical_inner_path, "member_checksum": member_checksum, "file_family": family},
                    )
                self.seen_member_checksums[member_checksum] = member_id
            emitter.archive_member(
                {
                    "run_id": parent.run_id,
                    "archive_member_id": member_id,
                    "archive_raw_asset_id": parent.raw_asset_id,
                    "source_id": parent.source_id,
                    "compound": parent.compound,
                    "archive_storage_path": parent.storage_path,
                    "archive_checksum": parent.checksum,
                    "inner_path": logical_inner_path,
                    "member_checksum": member_checksum,
                    "compressed_size": compressed_size,
                    "uncompressed_size": byte_count,
                    "expected_uncompressed_size": expected_size,
                    "file_extension": extension,
                    "file_family": family,
                    "parser_name": parser_name,
                    "parser_version": PARSER_VERSION,
                    "terminal_status": status,
                    "duplicate_target_raw_asset_id": duplicate_target,
                    "duplicate_target_archive_member_id": duplicate_member,
                    "source_location": emitter.location(parent, f"inner_path={logical_inner_path}"),
                    "warnings": warnings,
                    "metadata_json": dispatch_metadata,
                }
            )
            return status
        except Exception as exc:
            emitter.failure(
                parent,
                parser_name="archive_parser",
                failure_type="archive_member_read_failed",
                error=str(exc),
                source_location=emitter.location(parent, f"inner_path={inner_path}"),
                payload={"inner_path": inner_path},
            )
            return "failed"
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _write_rejected_member(
        self,
        asset: ParseAsset,
        emitter: ParseEmitter,
        inner_path: str,
        compressed_size: int | None,
        uncompressed_size: int,
        reason: str,
        failure_type: str = "unsafe_or_oversized_archive_member",
    ) -> None:
        emitter.archive_member(
            {
                "run_id": asset.run_id,
                "archive_member_id": stable_id("archive_member", asset.raw_asset_id, inner_path, reason),
                "archive_raw_asset_id": asset.raw_asset_id,
                "source_id": asset.source_id,
                "compound": asset.compound,
                "archive_storage_path": asset.storage_path,
                "archive_checksum": asset.checksum,
                "inner_path": inner_path,
                "member_checksum": None,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "file_extension": Path(inner_path).suffix.lower(),
                "file_family": None,
                "parser_name": "archive_parser",
                "parser_version": PARSER_VERSION,
                "terminal_status": "failed",
                "source_location": emitter.location(asset, f"inner_path={inner_path}"),
                "warnings": [reason],
            }
        )
        emitter.failure(
            asset,
            parser_name="archive_parser",
            failure_type=failure_type,
            error=reason,
            source_location=emitter.location(asset, f"inner_path={inner_path}"),
            payload={"inner_path": inner_path, "uncompressed_size": uncompressed_size},
        )


class RawParseStage:
    def __init__(
        self,
        *,
        run_root: Path,
        output_dir: Path,
        max_assets: int | None = None,
    ) -> None:
        self.run_root = run_root.resolve()
        self.run_id = self.run_root.name
        self.output_dir = output_dir.resolve()
        self.layout = raw_processing_layout(self.output_dir)
        self.max_assets = max_assets

    def run(self) -> dict[str, Any]:
        inventory_summary, pipeline_state = self._validate_prerequisite()
        assets = self._load_inventory()
        expected_asset_count = int(inventory_summary["raw_asset_count"])
        if len(assets) != expected_asset_count:
            raise ValueError(f"Inventory row count changed: expected={expected_asset_count} actual={len(assets)}")

        registry = ParserRegistry()
        RawFormatParsers().register(registry)
        standalone_checksums: dict[str, str] = {}
        for asset in assets:
            standalone_checksums.setdefault(asset.checksum, asset.raw_asset_id)
        staging_root = self.output_dir / "00_run_manifest" / ".archive_staging"
        archive_parser = ArchiveParser(registry, standalone_checksums, staging_root)
        registry.register("archive", "archive_parser", archive_parser.parse)

        canonical_by_key: dict[tuple[str, str, str], ParseAsset] = {}
        processed = 0
        progress_path = self.layout.path("parse_progress.json", for_write=True)
        write_json(
            progress_path,
            {
                "stage": "parse",
                "status": "in_progress",
                "run_id": self.run_id,
                "expected_raw_asset_count": expected_asset_count,
                "processed_raw_asset_count": 0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        with ParseEmitter(self.layout, self.run_id) as emitter:
            for asset in assets:
                if self.max_assets is not None and processed >= self.max_assets:
                    break
                processed += 1
                if asset.asset_metadata_json.get("processing_policy") == "metadata_only":
                    emitter.status(asset, parser_name="source_availability_gate", status="metadata_only",
                        warnings=[str(asset.asset_metadata_json.get("processing_policy_reason") or "source_availability_limit")],
                        metadata={"observation_extraction_blocked": True})
                    self._maybe_write_progress(progress_path, processed, expected_asset_count, asset, emitter)
                    continue
                dispatch_asset, dispatch_warnings, dispatch_metadata = prepare_asset_for_dispatch(asset)
                try:
                    registered = registry.resolve(dispatch_asset)
                except KeyError as exc:
                    emitter.failure(asset, parser_name="unregistered", failure_type="parser_not_registered", error=str(exc))
                    emitter.status(
                        asset,
                        parser_name="unregistered",
                        status="failed",
                        warnings=[*dispatch_warnings, str(exc)],
                        metadata=dispatch_metadata,
                    )
                    self._maybe_write_progress(progress_path, processed, expected_asset_count, asset, emitter)
                    continue
                dedup_key = (asset.checksum, registered.name, PARSER_VERSION)
                canonical = canonical_by_key.get(dedup_key)
                if canonical is not None:
                    emitter.alias(
                        {
                            "run_id": self.run_id,
                            "alias_raw_asset_id": asset.raw_asset_id,
                            "canonical_raw_asset_id": canonical.raw_asset_id,
                            "checksum": asset.checksum,
                            "file_family": dispatch_asset.file_family,
                            "parser_name": registered.name,
                            "parser_version": PARSER_VERSION,
                            "alias_source_id": asset.source_id,
                            "alias_compound": asset.compound,
                            "alias_request_id": asset.request_id,
                            "alias_request_key": asset.request_key,
                            "alias_source_url": asset.source_url,
                            "alias_storage_path": asset.storage_path,
                        }
                    )
                    emitter.status(
                        asset,
                        parser_name=registered.name,
                        status="duplicate_linked",
                        warnings=dispatch_warnings,
                        metadata={"canonical_raw_asset_id": canonical.raw_asset_id, **dispatch_metadata},
                    )
                    self._maybe_write_progress(progress_path, processed, expected_asset_count, asset, emitter)
                    continue
                try:
                    outcome = registered.callback(dispatch_asset, emitter)
                    emitter.status(
                        asset,
                        parser_name=registered.name,
                        status=outcome.status,
                        warnings=[*dispatch_warnings, *outcome.warnings],
                        metadata={**dispatch_metadata, **outcome.metadata},
                    )
                    # Failed bytes cannot serve as a successfully resolved
                    # canonical asset.  Parse later identical failures again so
                    # every source identity is retained in the failure ledger.
                    if outcome.status != "failed":
                        canonical_by_key[dedup_key] = asset
                except Exception as exc:
                    emitter.failure(
                        asset,
                        parser_name=registered.name,
                        failure_type="parse_exception",
                        error=str(exc),
                    )
                    emitter.status(
                        asset,
                        parser_name=registered.name,
                        status="failed",
                        warnings=[*dispatch_warnings, str(exc)],
                        metadata=dispatch_metadata,
                    )

                self._maybe_write_progress(progress_path, processed, expected_asset_count, asset, emitter)

            summary = self._build_summary(
                emitter,
                expected_asset_count=expected_asset_count,
                processed_asset_count=processed,
                canonical_asset_count=len(canonical_by_key),
                registry=registry,
            )

        if staging_root.exists():
            try:
                staging_root.rmdir()
            except OSError:
                pass
        link_counts = resolve_asset_links(self.layout, assets)
        summary["asset_link_resolution"] = link_counts
        provenance = audit_parse_provenance(self.layout)
        summary["provenance_audit"] = provenance
        summary["accepted"] = bool(
            summary["asset_parse_status_count"] == expected_asset_count
            and summary["unhandled_raw_asset_count"] == 0
            and summary["failed_asset_count"] == 0
            and summary["processing_failure_count"] == 0
            and provenance["missing_required_provenance"] == 0
        )
        summary["status"] = "completed" if summary["accepted"] else "failed"
        summary["generated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(
            progress_path,
            {
                "stage": "parse",
                "status": "completed" if summary["accepted"] else "failed",
                "run_id": self.run_id,
                "expected_raw_asset_count": expected_asset_count,
                "processed_raw_asset_count": summary["asset_parse_status_count"],
                "terminal_status_counts": summary["terminal_status_counts"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        write_json(self.layout.path("parse_summary.json", for_write=True), summary)
        update_pipeline_state(
            self.layout.path("pipeline_state.json", for_write=True),
            pipeline_state,
            summary,
            self.layout.path("parse_summary.json", for_write=True),
        )
        return summary

    def _validate_prerequisite(self) -> tuple[dict[str, Any], dict[str, Any]]:
        inventory_path = self.layout.path("raw_inventory.jsonl")
        summary_path = self.layout.path("inventory_summary.json")
        state_path = self.layout.path("pipeline_state.json")
        for path in [inventory_path, summary_path, state_path]:
            if not path.is_file():
                raise FileNotFoundError(f"Parse prerequisite missing: {path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        inventory_state = (state.get("stages") or {}).get("inventory") or {}
        if summary.get("accepted") is not True or summary.get("status") != "completed":
            raise ValueError("Inventory summary is not accepted")
        if inventory_state.get("status") != "completed" or inventory_state.get("accepted") is not True:
            raise ValueError("pipeline_state inventory prerequisite is not accepted")
        return summary, state

    def _load_inventory(self) -> list[ParseAsset]:
        rows = []
        with self.layout.path("raw_inventory.jsonl").open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(ParseAsset.from_inventory(json.loads(line)))
        return rows

    def _build_summary(
        self,
        emitter: ParseEmitter,
        *,
        expected_asset_count: int,
        processed_asset_count: int,
        canonical_asset_count: int,
        registry: ParserRegistry,
    ) -> dict[str, Any]:
        status_count = emitter.counts["asset_parse_status"]
        return {
            "stage": "parse",
            "run_id": self.run_id,
            "run_root": str(self.run_root),
            "output_dir": str(self.output_dir),
            "parser_version": PARSER_VERSION,
            "registered_parsers": registry.registered_keys(),
            "expected_raw_asset_count": expected_asset_count,
            "processed_raw_asset_count": processed_asset_count,
            "asset_parse_status_count": status_count,
            "unhandled_raw_asset_count": max(0, expected_asset_count - status_count),
            "canonical_asset_count": canonical_asset_count,
            "alias_asset_count": emitter.counts["raw_asset_aliases"],
            "terminal_status_counts": dict(sorted(emitter.status_counts.items())),
            "failed_asset_count": emitter.status_counts["failed"],
            "unsupported_asset_count": emitter.status_counts["unsupported"],
            "parser_counts": dict(sorted(emitter.parser_counts.items())),
            "parsed_document_count": emitter.counts["parsed_documents"],
            "parsed_record_count": emitter.counts["parsed_records"],
            "parsed_table_count": emitter.counts["parsed_tables"],
            "text_chunk_count": emitter.counts["text_chunks"],
            "archive_member_count": emitter.counts["archive_members"],
            "asset_link_count": emitter.counts["asset_links"],
            "processing_failure_count": emitter.counts["processing_failures"],
            "failure_counts": dict(sorted(emitter.failure_counts.items())),
            "evidence_rows_created": 0,
            "llm_requests_created": 0,
            "embeddings_created": 0,
        }

    def _maybe_write_progress(
        self,
        path: Path,
        processed: int,
        expected: int,
        asset: ParseAsset,
        emitter: ParseEmitter,
    ) -> None:
        if processed % 100 != 0 and processed != expected and (
            self.max_assets is None or processed != self.max_assets
        ):
            return
        write_json(
            path,
            {
                "stage": "parse",
                "status": "in_progress",
                "run_id": self.run_id,
                "expected_raw_asset_count": expected,
                "processed_raw_asset_count": processed,
                "current_raw_asset_id": asset.raw_asset_id,
                "current_storage_path": asset.storage_path,
                "terminal_status_counts": dict(sorted(emitter.status_counts.items())),
                "output_counts": dict(sorted(emitter.counts.items())),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )


def is_rar_archive(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            signature = file.read(8)
    except OSError:
        return False
    return signature.startswith((b"Rar!\x1a\x07\x00", b"Rar!\x1a\x07\x01\x00"))


def parse_bsdtar_verbose_listing(output: str) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=8)
        if len(parts) != 9:
            raise ValueError(f"Unexpected bsdtar listing at line {line_number}: {line[:200]}")
        try:
            size = int(parts[4])
        except ValueError as exc:
            raise ValueError(f"Invalid bsdtar member size at line {line_number}: {parts[4]}") from exc
        members.append({"mode": parts[0], "size": size, "inner_path": parts[8]})
    return members


def archive_path_rejection(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or normalized.startswith("/"):
        return "absolute_path"
    if ".." in path.parts:
        return "path_traversal"
    if not normalized or normalized.endswith("/"):
        return "empty_or_directory"
    return None


def normalize_resource_name(value: str) -> str:
    name = Path(value.split("?", 1)[0]).name.lower()
    stem = re.sub(r"_[0-9a-f]{12,64}(?=\.[^.]+$)", "", name)
    return stem


def extract_pmcid(path: str) -> str | None:
    match = re.search(r"\bPMC\d+\b", path, re.I)
    return match.group(0).upper() if match else None


def resolve_asset_links(layout: RawProcessingLayout, assets: list[ParseAsset]) -> dict[str, int]:
    path = layout.path("asset_links.jsonl", for_write=True)
    if not path.exists():
        path.touch()
        return {"total": 0, "resolved": 0, "unresolved": 0, "ambiguous": 0}
    targets: dict[tuple[str | None, str], list[ParseAsset]] = {}
    assets_by_id = {asset.raw_asset_id: asset for asset in assets}
    alias_to_canonical: dict[str, str] = {}
    alias_path = layout.path("raw_asset_aliases.jsonl", for_write=True)
    if alias_path.is_file():
        with alias_path.open("r", encoding="utf-8") as alias_file:
            for line in alias_file:
                if not line.strip():
                    continue
                alias = json.loads(line)
                alias_to_canonical[str(alias["alias_raw_asset_id"])] = str(alias["canonical_raw_asset_id"])
    for asset in assets:
        if asset.file_family not in {"image", "media", "document", "presentation", "table", "pdf", "text", "html"}:
            continue
        pmcid = extract_pmcid(asset.storage_path)
        keys = {normalize_resource_name(Path(asset.storage_path).name), Path(normalize_resource_name(Path(asset.storage_path).name)).stem}
        for key in keys:
            targets.setdefault((pmcid, key), []).append(asset)
    rewritten: list[dict[str, Any]] = []
    resolution_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            href = str(row.get("target_href") or "")
            pmcid = str((row.get("metadata_json") or {}).get("pmcid") or "").upper() or None
            names = [normalize_resource_name(href), Path(normalize_resource_name(href)).stem]
            matches: list[ParseAsset] = []
            for name in names:
                matches.extend(targets.get((pmcid, name), []))
            unique = {match.raw_asset_id: match for match in matches}
            for field in [
                "target_raw_asset_id",
                "target_checksum",
                "candidate_target_raw_asset_ids",
                "resolution_method",
            ]:
                row.pop(field, None)
            if len(unique) == 1:
                target = next(iter(unique.values()))
                row["target_raw_asset_id"] = target.raw_asset_id
                row["target_checksum"] = target.checksum
                row["resolution_status"] = "resolved"
                row["resolution_method"] = "single_normalized_name_match"
                resolution_counts["resolved"] += 1
            elif unique and len({target.checksum for target in unique.values()}) == 1:
                canonical_ids = {
                    alias_to_canonical.get(raw_asset_id, raw_asset_id)
                    for raw_asset_id in unique
                }
                target_id = sorted(canonical_ids)[0]
                target = assets_by_id.get(target_id) or unique[sorted(unique)[0]]
                row["target_raw_asset_id"] = target.raw_asset_id
                row["target_checksum"] = target.checksum
                row["candidate_target_raw_asset_ids"] = sorted(unique)
                row["resolution_status"] = "resolved"
                row["resolution_method"] = "identical_checksum_canonical"
                resolution_counts["resolved"] += 1
            else:
                row["resolution_status"] = "unresolved" if not unique else "ambiguous"
                row["candidate_target_raw_asset_ids"] = sorted(unique)
                resolution_counts[row["resolution_status"]] += 1
            rewritten.append(row)
    with path.open("w", encoding="utf-8") as file:
        for row in rewritten:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")
    return {
        "total": len(rewritten),
        "resolved": resolution_counts["resolved"],
        "unresolved": resolution_counts["unresolved"],
        "ambiguous": resolution_counts["ambiguous"],
    }


def audit_parse_provenance(layout: RawProcessingLayout) -> dict[str, int]:
    missing = 0
    row_count = 0
    for name in ["parsed_documents.jsonl", "parsed_records.jsonl", "parsed_tables.jsonl", "text_chunks.jsonl"]:
        path = layout.path(name, for_write=True)
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row_count += 1
                row = json.loads(line)
                if not row.get("raw_asset_id") or not row.get("checksum") or not row.get("source_location"):
                    missing += 1
    return {"audited_rows": row_count, "missing_required_provenance": missing}


def update_pipeline_state(path: Path, state: dict[str, Any], summary: dict[str, Any], summary_path: Path) -> None:
    stages = state.setdefault("stages", {})
    stages["parse"] = {
        "status": "completed" if summary["accepted"] else "failed",
        "accepted": summary["accepted"],
        "summary_path": str(summary_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    for downstream in ["deterministic_extract", "llm_extract", "completeness", "embedding"]:
        stages[downstream] = {"status": "not_started"}
    state["current_stage"] = "parse"
    write_json(path, state)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
