from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from pbpk_kb.connectors.models import ConnectorRunLog, ParsedSourceRecord, RawDocumentMetadata


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def checksum_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


class ObjectStore:
    """Versioned file storage for raw source payloads, parsed records, and logs."""

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.raw_root = self.root / "raw"
        self.parsed_root = self.root / "parsed"
        self.log_root = self.root / "ingestion_logs"
        for path in (self.raw_root, self.parsed_root, self.log_root):
            path.mkdir(parents=True, exist_ok=True)

    def write_raw_snapshot(
        self,
        *,
        source_id: str,
        ingestion_run_id: str,
        source_url: str,
        payload: bytes,
        extension: str,
        content_type: str | None,
        status_code: int | None,
        raw_record_count: int,
        source_version: str | None,
        is_sample_placeholder: bool,
        retrieval_method: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RawDocumentMetadata:
        download_timestamp = utc_now()
        checksum = checksum_bytes(payload)
        safe_extension = extension.lstrip(".") or "bin"
        filename = f"{download_timestamp.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex}.{safe_extension}"
        run_dir = self.raw_root / source_id / ingestion_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        raw_path = run_dir / filename
        raw_path.write_bytes(payload)

        metadata = RawDocumentMetadata(
            source_id=source_id,
            source_url=source_url,
            download_timestamp=download_timestamp,
            raw_checksum=checksum,
            raw_storage_path=str(raw_path),
            raw_metadata_path=str(raw_path.with_suffix(raw_path.suffix + ".metadata.json")),
            content_type=content_type,
            status_code=status_code,
            raw_record_count=raw_record_count,
            source_version=source_version,
            is_sample_placeholder=is_sample_placeholder,
            retrieval_method=retrieval_method,
            extra_metadata=extra_metadata or {},
        )
        self.write_json(raw_path.with_suffix(raw_path.suffix + ".metadata.json"), metadata.model_dump(mode="json"))
        return metadata

    def write_parsed_records(
        self,
        *,
        source_id: str,
        ingestion_run_id: str,
        records: list[ParsedSourceRecord],
    ) -> Path:
        run_dir = self.parsed_root / source_id / ingestion_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        records_path = run_dir / "records.jsonl"
        with records_path.open("w", encoding="utf-8") as records_file:
            for record in records:
                records_file.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
                records_file.write("\n")
        self.write_json(
            run_dir / "records.metadata.json",
            {
                "source_id": source_id,
                "ingestion_run_id": ingestion_run_id,
                "parsed_record_count": len(records),
                "created_at": utc_now().isoformat(),
            },
        )
        return records_path

    def write_ingestion_log(self, log: ConnectorRunLog) -> Path:
        log_dir = self.log_root / log.source_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{log.ingestion_run_id}.json"
        self.write_json(log_path, log.model_dump(mode="json"))
        latest_path = log_dir / "latest.json"
        self.write_json(latest_path, log.model_dump(mode="json"))
        return log_path

    @staticmethod
    def write_json(path: str | Path, payload: Any) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
            json_file.write("\n")
