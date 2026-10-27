"""Resumable transfer of the official PMC static-object download plan.

The acquisition finalizer inventories every selected PMC OA object, but it is
intentionally constrained for short metadata requests.  This module performs
the long-running raw-file transfer separately so that large PDFs, XML files,
figures, and supplements are streamed to disk, resumable, and auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from uuid import uuid4

from pbpk_kb.acquisition.comprehensive_raw_retriever import append_jsonl, iso_now, write_json


MAX_RANGE_REQUEST_BYTES = 1024 * 1024


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def object_key(row: dict[str, Any]) -> str | None:
    """Return the stable plan identity for either S3 or legacy PMC objects."""

    s3_key = row.get("s3_key")
    if s3_key:
        return f"s3:{s3_key}"
    url = row.get("url")
    return f"url:{url}" if url else None


def object_key_from_asset(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    s3_key = metadata.get("s3_key")
    if s3_key:
        return f"s3:{s3_key}"
    url = row.get("url")
    return f"url:{url}" if url else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - only compares a source-provided S3 ETag.
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unknown"


def file_suffix(row: dict[str, Any]) -> str:
    source = str(row.get("s3_key") or urlparse(str(row.get("url") or "")).path)
    name = Path(source).name
    if name.lower().endswith(".tar.gz"):
        return ".tar.gz"
    suffix = Path(name).suffix
    return suffix if suffix else ".bin"


def content_priority(row: dict[str, Any]) -> int:
    """Order raw PMC objects by evidence value before incidental media."""

    suffix = file_suffix(row).lower()
    if suffix == ".xml":
        return 0
    if suffix in {".txt", ".json", ".html"}:
        return 1
    if suffix == ".pdf":
        return 2
    if suffix in {".csv", ".tsv", ".xls", ".xlsx"}:
        return 3
    if suffix in {".doc", ".docx", ".ppt", ".pptx", ".zip", ".rar", ".tar.gz"}:
        return 4
    if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".eps", ".avi", ".mp3", ".mp4"}:
        return 6
    return 5


def storage_path(run_root: Path, row: dict[str, Any]) -> Path:
    identity = object_key(row)
    if identity is None:
        raise ValueError("PMC object plan row has no s3_key or URL")
    object_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    pmcid = safe_component(str(row.get("pmcid") or "unknown_pmcid"))
    name = Path(str(row.get("s3_key") or urlparse(str(row["url"])).path)).name
    suffix = file_suffix(row)
    if name.lower().endswith(".tar.gz"):
        stem = name[: -len(".tar.gz")]
    else:
        stem = Path(name).stem
    return run_root / "shared" / "pmc_static_objects" / pmcid / "pmc_oa" / f"{safe_component(stem or 'pmc_object')}_{object_id}{suffix}"


def expected_bytes(row: dict[str, Any]) -> int | None:
    value = row.get("byte_count")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


@dataclass(frozen=True)
class TransferResult:
    key: str
    status: str
    bytes_downloaded: int
    expected_bytes: int | None
    storage_path: str | None = None
    checksum: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ByteRange:
    """One inclusive HTTP byte range within a planned PMC object."""

    start: int
    end: int

    @property
    def byte_count(self) -> int:
        return self.end - self.start + 1


def split_byte_ranges(
    *,
    start_offset: int,
    expected_bytes: int,
    connections: int,
    min_segment_bytes: int = 1024 * 1024,
) -> list[ByteRange]:
    """Split the un-downloaded suffix into balanced, non-overlapping ranges."""

    if expected_bytes <= start_offset:
        return []
    remaining = expected_bytes - start_offset
    segments = min(max(1, connections), max(1, remaining // max(1, min_segment_bytes)))
    base_size, extra = divmod(remaining, segments)
    ranges: list[ByteRange] = []
    cursor = start_offset
    for index in range(segments):
        length = base_size + (1 if index < extra else 0)
        ranges.append(ByteRange(start=cursor, end=cursor + length - 1))
        cursor += length
    return ranges


def fixed_byte_ranges(expected_bytes: int, chunk_bytes: int = MAX_RANGE_REQUEST_BYTES) -> list[ByteRange]:
    """Return contiguous fixed-size ranges covering one complete object."""

    if expected_bytes <= 0:
        return []
    ranges: list[ByteRange] = []
    start = 0
    while start < expected_bytes:
        end = min(expected_bytes - 1, start + chunk_bytes - 1)
        ranges.append(ByteRange(start=start, end=end))
        start = end + 1
    return ranges


class PMCObjectTransfer:
    def __init__(
        self,
        *,
        run_root: Path,
        workers: int,
        connect_timeout_seconds: int,
        low_speed_limit_bytes: int,
        low_speed_time_seconds: int,
        max_transfer_seconds: int,
        retries: int,
        range_connections: int,
        range_min_bytes: int,
        max_connections: int,
        dry_run: bool,
        max_objects: int | None,
    ) -> None:
        self.run_root = run_root
        self.plan_path = run_root / "finalization" / "pmc_object_download_plan.jsonl"
        self.assets_path = run_root / "assets.jsonl"
        self.transfer_root = run_root / "finalization" / "pmc_object_transfer"
        self.ledger_path = self.transfer_root / "transfer.jsonl"
        self.status_path = self.transfer_root / "STATUS.json"
        self.workers = max(1, workers)
        self.connect_timeout_seconds = connect_timeout_seconds
        self.low_speed_limit_bytes = low_speed_limit_bytes
        self.low_speed_time_seconds = low_speed_time_seconds
        self.max_transfer_seconds = max(1, max_transfer_seconds)
        self.retries = retries
        self.range_connections = max(1, range_connections)
        self.range_min_bytes = max(1, range_min_bytes)
        self.max_connections = max(1, max_connections)
        self.connection_slots = threading.BoundedSemaphore(self.max_connections)
        self.dry_run = dry_run
        self.max_objects = max_objects
        self.write_lock = threading.Lock()

    def plan_rows(self) -> list[dict[str, Any]]:
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for row in read_jsonl(self.plan_path):
            key = object_key(row)
            if key and key not in seen:
                seen.add(key)
                rows.append(row)
        return rows

    def completed_assets(self, plan_keys: set[str]) -> dict[str, dict[str, Any]]:
        completed: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(self.assets_path):
            if row.get("source_id") != "pmc_oa":
                continue
            key = object_key_from_asset(row)
            path = Path(str(row.get("storage_path") or ""))
            recorded_bytes = int(row.get("byte_count") or -1)
            if key in plan_keys and path.is_file() and path.stat().st_size == recorded_bytes:
                completed[key] = row
        return completed

    def run(self) -> dict[str, Any]:
        if not self.plan_path.exists():
            raise FileNotFoundError(f"PMC object download plan does not exist: {self.plan_path}")

        plan = self.plan_rows()
        plan_keys = {key for row in plan if (key := object_key(row))}
        completed = self.completed_assets(plan_keys)
        pending = [row for row in plan if object_key(row) not in completed]
        # Prioritize source text and PDFs before supplementary media. Every
        # planned object remains in the queue and later media is still fetched.
        pending.sort(
            key=lambda row: (
                content_priority(row),
                expected_bytes(row) is None,
                expected_bytes(row) or 0,
                object_key(row) or "",
            )
        )
        limited_run = self.max_objects is not None and len(pending) > self.max_objects
        if self.max_objects is not None:
            pending = pending[: self.max_objects]

        self.transfer_root.mkdir(parents=True, exist_ok=True)
        started_at = iso_now()
        self.write_status("running", plan, completed, pending, started_at=started_at)

        for key, asset in completed.items():
            self.log_event(
                {
                    "status": "already_available",
                    "object_key": key,
                    "asset_id": asset.get("asset_id"),
                    "storage_path": asset.get("storage_path"),
                    "recorded_at": iso_now(),
                }
            )

        results: list[TransferResult] = []
        if self.dry_run:
            results = [
                TransferResult(
                    key=object_key(row) or "unknown",
                    status="dry_run_pending",
                    bytes_downloaded=0,
                    expected_bytes=expected_bytes(row),
                )
                for row in pending
            ]
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self.transfer_one, row): row for row in pending}
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    self.log_terminal_result(result)
                    self.write_status("running", plan, completed, pending, started_at=started_at, results=results)

        if self.dry_run:
            for result in results:
                self.log_terminal_result(result)

        failed = [result for result in results if result.status == "failed"]
        if self.dry_run:
            terminal = "dry_run_completed"
        elif failed:
            terminal = "completed_with_failures"
        elif limited_run:
            terminal = "partial_completed"
        else:
            terminal = "completed"
        status = self.write_status(terminal, plan, completed, pending, started_at=started_at, results=results)
        if terminal == "completed":
            from pbpk_kb.acquisition.reconcile_final_acquisition import reconcile

            download_plan_path = self.run_root / "download_plan.json"
            download_plan = json.loads(download_plan_path.read_text(encoding="utf-8")) if download_plan_path.exists() else {}
            compounds = [str(value) for value in download_plan.get("compounds", []) if str(value)]
            output_root = self.run_root.parents[1] if self.run_root.parent.name == "compound_runs" else self.run_root.parent
            final_result = reconcile(self.run_root, output_root, compounds)
            status.update(
                {
                    "final_result_status": final_result.get("status"),
                    "accepted_as_complete_raw_content": final_result.get("accepted_as_complete_raw_content"),
                    "final_acquisition_result_path": str(
                        self.run_root / "finalization" / "FINAL_ACQUISITION_RESULT.json"
                    ),
                }
            )
            write_json(self.status_path, status)
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
        return status

    def transfer_one(self, row: dict[str, Any]) -> TransferResult:
        key = object_key(row)
        if key is None:
            return TransferResult("unknown", "failed", 0, expected_bytes(row), error="Plan row has no object key")
        url = str(row.get("url") or "")
        if not url:
            return TransferResult(key, "failed", 0, expected_bytes(row), error="Plan row has no download URL")
        final_path = storage_path(self.run_root, row)
        temporary_path = final_path.with_suffix(final_path.suffix + ".part")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        expected = expected_bytes(row)

        try:
            self.log_event({"status": "started", "object_key": key, "url": url, "recorded_at": iso_now()})
            self.download_with_resume(url, temporary_path, expected)
            actual = temporary_path.stat().st_size
            if expected is not None and actual != expected:
                raise RuntimeError(f"Downloaded size mismatch: expected {expected}, received {actual}")
            checksum = sha256_file(temporary_path)
            etag = str(row.get("etag") or "").strip('"')
            if re.fullmatch(r"[0-9a-fA-F]{32}", etag) and md5_file(temporary_path).lower() != etag.lower():
                raise RuntimeError("Downloaded file does not match the source ETag")
            os.replace(temporary_path, final_path)
            asset = self.record_asset(row, key, final_path, checksum, actual)
            return TransferResult(key, "completed", actual, expected, str(final_path), checksum)
        except Exception as exc:
            actual = temporary_path.stat().st_size if temporary_path.exists() else 0
            return TransferResult(key, "failed", actual, expected, str(temporary_path) if temporary_path.exists() else None, error=str(exc))

    def curl_base_command(self) -> list[str]:
        return [
            "curl",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            str(self.retries),
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--connect-timeout",
            str(self.connect_timeout_seconds),
            "--speed-limit",
            str(self.low_speed_limit_bytes),
            "--speed-time",
            str(self.low_speed_time_seconds),
            "--max-time",
            str(self.max_transfer_seconds),
            "-A",
            "pbpk-public-pmc-static-transfer/0.2",
        ]

    def run_curl(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        # Large objects can use several range requests, so cap all curl
        # processes across workers rather than multiplying connections freely.
        with self.connection_slots:
            return subprocess.run(command, text=True, capture_output=True)

    def download_with_resume(self, url: str, temporary_path: Path, expected: int | None) -> None:
        if expected and expected >= self.range_min_bytes and self.range_connections > 1:
            self.download_with_verified_chunks(url, temporary_path, expected)
            return
        self.download_with_curl_resume(url, temporary_path)

    def download_with_curl_resume(self, url: str, temporary_path: Path) -> None:
        command = [
            *self.curl_base_command(),
            "--continue-at",
            "-",
            "--output",
            str(temporary_path),
            url,
        ]
        result = self.run_curl(command)
        if result.returncode == 0:
            return
        # A few legacy hosts do not support Range requests. Restart once from
        # zero instead of leaving an unusable partial file.
        if temporary_path.exists() and "range" in (result.stderr or "").lower():
            temporary_path.unlink()
            command.remove("--continue-at")
            command.remove("-")
            result = self.run_curl(command)
            if result.returncode == 0:
                return
        raise RuntimeError((result.stderr or result.stdout or f"curl exited {result.returncode}").strip())

    def range_state_path(self, temporary_path: Path) -> Path:
        return temporary_path.with_name(f"{temporary_path.name}.ranges.json")

    def prepare_range_state(self, url: str, temporary_path: Path, expected: int) -> dict[str, Any]:
        prefix_bytes = temporary_path.stat().st_size if temporary_path.exists() else 0
        if prefix_bytes > expected:
            temporary_path.unlink()
            prefix_bytes = 0
        state_path = self.range_state_path(temporary_path)
        existing_state: dict[str, Any] | None = None
        if state_path.exists():
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                existing_state = loaded

        if (
            existing_state
            and existing_state.get("url") == url
            and existing_state.get("expected_bytes") == expected
            and existing_state.get("prefix_bytes") == prefix_bytes
            and isinstance(existing_state.get("segments"), list)
        ):
            return existing_state

        for path in temporary_path.parent.glob(f"{temporary_path.name}.range*"):
            path.unlink(missing_ok=True)
        segments = []
        for index, byte_range in enumerate(
            split_byte_ranges(
                start_offset=prefix_bytes,
                expected_bytes=expected,
                connections=self.range_connections,
            )
        ):
            segment_path = temporary_path.with_name(f"{temporary_path.name}.range{index:02d}")
            segments.append(
                {
                    "start": byte_range.start,
                    "end": byte_range.end,
                    "path": str(segment_path),
                }
            )
        state = {
            "url": url,
            "expected_bytes": expected,
            "prefix_bytes": prefix_bytes,
            "segments": segments,
            "created_at": iso_now(),
        }
        write_json(state_path, state)
        return state

    def download_range(self, url: str, segment: dict[str, Any]) -> None:
        path = Path(str(segment["path"]))
        start = int(segment["start"])
        end = int(segment["end"])
        expected = end - start + 1
        existing = path.stat().st_size if path.exists() else 0
        if existing > expected:
            path.unlink()
            existing = 0
        # Some proxies acknowledge a partial HTTP 206 response as successful
        # even when they return fewer bytes than the requested range. Continue
        # from the verified local size until this segment is actually complete.
        max_requests = max(8, min(512, (expected + 262_143) // 262_144 + 4))
        request_count = 0
        no_progress_responses = 0
        while existing < expected:
            request_start = start + existing
            # The active VPN reliably serves 1 MiB Range responses but can
            # silently truncate larger ranges. Keep each HTTP request bounded
            # and stitch the verified responses into the logical segment.
            request_end = min(end, request_start + MAX_RANGE_REQUEST_BYTES - 1)
            command = [
                *self.curl_base_command(),
                "--range",
                f"{request_start}-{request_end}",
                "--append",
                "--output",
                str(path),
                url,
            ]
            result = self.run_curl(command)
            actual = path.stat().st_size if path.exists() else 0
            if result.returncode != 0:
                raise RuntimeError((result.stderr or result.stdout or f"curl exited {result.returncode}").strip())
            request_count += 1
            if actual <= existing:
                no_progress_responses += 1
                if no_progress_responses >= 6 or request_count >= max_requests:
                    raise RuntimeError(
                        f"Range request made no progress after {no_progress_responses} retries: "
                        f"expected {expected}, received {actual}"
                    )
                time.sleep(min(1.0, 0.25 * no_progress_responses))
                continue
            no_progress_responses = 0
            existing = actual
            if request_count >= max_requests and existing < expected:
                raise RuntimeError(
                    f"Range remained incomplete after {request_count} responses: "
                    f"expected {expected}, received {existing}"
                )

    def download_with_parallel_ranges(self, url: str, temporary_path: Path, expected: int) -> None:
        state = self.prepare_range_state(url, temporary_path, expected)
        segments = state["segments"]
        if not segments:
            return
        with ThreadPoolExecutor(max_workers=min(self.range_connections, len(segments))) as executor:
            futures = [executor.submit(self.download_range, url, segment) for segment in segments]
            for future in as_completed(futures):
                future.result()

        assembled_path = temporary_path.with_name(f"{temporary_path.name}.assemble")
        with assembled_path.open("wb") as destination:
            if temporary_path.exists():
                with temporary_path.open("rb") as prefix:
                    shutil.copyfileobj(prefix, destination, 1024 * 1024)
            for segment in segments:
                with Path(str(segment["path"])).open("rb") as source:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
        if assembled_path.stat().st_size != expected:
            raise RuntimeError(f"Assembled size mismatch: expected {expected}, received {assembled_path.stat().st_size}")
        os.replace(assembled_path, temporary_path)
        for segment in segments:
            Path(str(segment["path"])).unlink(missing_ok=True)
        self.range_state_path(temporary_path).unlink(missing_ok=True)

    def chunk_path(self, temporary_path: Path, index: int) -> Path:
        return temporary_path.with_name(f"{temporary_path.name}.chunk{index:06d}")

    def download_verified_chunk(self, url: str, byte_range: ByteRange, path: Path) -> None:
        expected = byte_range.byte_count
        if path.is_file() and path.stat().st_size == expected:
            return
        path.unlink(missing_ok=True)
        partial_path = path.with_name(f"{path.name}.partial")
        partial_path.unlink(missing_ok=True)
        last_error = ""

        # A successful HTTP response may still be shorter than its requested
        # range when a proxy imposes a response-size cap. Only a fully sized
        # chunk becomes resumable state.
        for attempt in range(max(4, self.retries + 1)):
            command = [
                *self.curl_base_command(),
                "--range",
                f"{byte_range.start}-{byte_range.end}",
                "--output",
                str(partial_path),
                url,
            ]
            result = self.run_curl(command)
            actual = partial_path.stat().st_size if partial_path.exists() else 0
            if result.returncode == 0 and actual == expected:
                os.replace(partial_path, path)
                return
            last_error = (result.stderr or result.stdout or "").strip()
            if not last_error:
                last_error = f"Range size mismatch: expected {expected}, received {actual}"
            partial_path.unlink(missing_ok=True)
            if attempt + 1 < max(4, self.retries + 1):
                time.sleep(min(1.0, 0.25 * (attempt + 1)))
        raise RuntimeError(last_error)

    def download_with_verified_chunks(self, url: str, temporary_path: Path, expected: int) -> None:
        ranges = fixed_byte_ranges(expected)
        chunks = [(byte_range, self.chunk_path(temporary_path, index)) for index, byte_range in enumerate(ranges)]
        with ThreadPoolExecutor(max_workers=min(self.range_connections, len(chunks))) as executor:
            futures = [
                executor.submit(self.download_verified_chunk, url, byte_range, path)
                for byte_range, path in chunks
            ]
            for future in as_completed(futures):
                future.result()

        assembled_path = temporary_path.with_name(f"{temporary_path.name}.assemble")
        with assembled_path.open("wb") as destination:
            for _byte_range, path in chunks:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
        if assembled_path.stat().st_size != expected:
            raise RuntimeError(f"Assembled size mismatch: expected {expected}, received {assembled_path.stat().st_size}")
        os.replace(assembled_path, temporary_path)
        for _byte_range, path in chunks:
            path.unlink(missing_ok=True)
            path.with_name(f"{path.name}.partial").unlink(missing_ok=True)
        for legacy_path in temporary_path.parent.glob(f"{temporary_path.name}.range*"):
            legacy_path.unlink(missing_ok=True)
        self.range_state_path(temporary_path).unlink(missing_ok=True)

    def record_asset(self, row: dict[str, Any], key: str, path: Path, checksum: str, byte_count: int) -> dict[str, Any]:
        object_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        asset = {
            "asset_id": f"asset:{uuid4().hex}",
            "request_id": f"pmc_oa:static_object_{object_id}",
            "source_id": "pmc_oa",
            "compound": None,
            "group": f"pmc/oa_s3_objects/{safe_component(str(row.get('pmcid') or 'unknown_pmcid'))}",
            "request_key": f"static_object_{object_id}",
            "url": row.get("url"),
            "storage_path": str(path),
            "metadata_path": str(path.with_suffix(path.suffix + ".metadata.json")),
            "checksum": checksum,
            "byte_count": byte_count,
            "content_type": None,
            "status_code": 200,
            "downloaded_at": iso_now(),
            "transport_warning": None,
            "metadata": {
                "pmcid": row.get("pmcid"),
                "s3_bucket": row.get("s3_bucket"),
                "s3_key": row.get("s3_key"),
                "s3_etag": row.get("etag"),
                "s3_size": row.get("byte_count"),
                "plan_status": row.get("status"),
                "plan_reason": row.get("reason"),
                "origin": row.get("origin"),
                "transfer_method": "official_pmc_static_object_plan",
            },
            "raw_only": True,
        }
        write_json(Path(str(asset["metadata_path"])), asset)
        with self.write_lock:
            append_jsonl(self.assets_path, asset)
        return asset

    def log_event(self, event: dict[str, Any]) -> None:
        with self.write_lock:
            append_jsonl(self.ledger_path, event)

    def log_terminal_result(self, result: TransferResult) -> None:
        self.log_event(
            {
                "status": result.status,
                "object_key": result.key,
                "byte_count": result.bytes_downloaded,
                "expected_byte_count": result.expected_bytes,
                "storage_path": result.storage_path,
                "checksum": result.checksum,
                "error": result.error,
                "recorded_at": iso_now(),
            }
        )

    def write_status(
        self,
        state: str,
        plan: list[dict[str, Any]],
        completed: dict[str, dict[str, Any]],
        pending: list[dict[str, Any]],
        *,
        started_at: str,
        results: Iterable[TransferResult] = (),
    ) -> dict[str, Any]:
        result_list = list(results)
        completed_now = sum(1 for result in result_list if result.status == "completed")
        failed_now = sum(1 for result in result_list if result.status == "failed")
        planned_bytes = sum(expected_bytes(row) or 0 for row in plan)
        existing_bytes = sum(int(row.get("byte_count") or 0) for row in completed.values())
        completed_bytes = sum(result.bytes_downloaded for result in result_list if result.status == "completed")
        return_payload = {
            "status": state,
            "run_root": str(self.run_root),
            "plan_path": str(self.plan_path),
            "ledger_path": str(self.ledger_path),
            "started_at": started_at,
            "updated_at": iso_now(),
            "workers": self.workers,
            "range_connections": self.range_connections,
            "range_min_bytes": self.range_min_bytes,
            "range_request_bytes": MAX_RANGE_REQUEST_BYTES,
            "max_connections": self.max_connections,
            "plan_object_count": len(plan),
            "plan_bytes": planned_bytes,
            "already_available_object_count": len(completed),
            "already_available_bytes": existing_bytes,
            "attempted_object_count": len(pending),
            "completed_this_invocation_count": completed_now,
            "completed_this_invocation_bytes": completed_bytes,
            "failed_this_invocation_count": failed_now,
            "remaining_after_invocation_count": max(0, len(plan) - len(completed) - completed_now),
            "raw_only": True,
        }
        with self.write_lock:
            write_json(self.status_path, return_payload)
        return return_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume physical transfer of a PMC static-object plan.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--connect-timeout-seconds", type=int, default=30)
    parser.add_argument("--low-speed-limit-bytes", type=int, default=8192)
    parser.add_argument("--low-speed-time-seconds", type=int, default=120)
    parser.add_argument("--max-transfer-seconds", type=int, default=600)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--range-connections", type=int, default=4)
    parser.add_argument("--range-min-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-connections", type=int, default=16)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    PMCObjectTransfer(
        run_root=args.run_root,
        workers=args.workers,
        connect_timeout_seconds=args.connect_timeout_seconds,
        low_speed_limit_bytes=args.low_speed_limit_bytes,
        low_speed_time_seconds=args.low_speed_time_seconds,
        max_transfer_seconds=args.max_transfer_seconds,
        retries=args.retries,
        range_connections=args.range_connections,
        range_min_bytes=args.range_min_bytes,
        max_connections=args.max_connections,
        dry_run=args.dry_run,
        max_objects=args.max_objects,
    ).run()


if __name__ == "__main__":
    main()
