from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from pbpk_kb.config.sources import SourceRegistryEntry, load_source_registry
from pbpk_kb.connectors.models import (
    ConnectorError,
    ConnectorRunLog,
    ConnectorRunResult,
    ParsedSourceRecord,
    RawDocumentMetadata,
)
from pbpk_kb.storage.object_store import ObjectStore, json_bytes, utc_now


class LiveFetchUnavailable(RuntimeError):
    pass


class BaseConnector:
    source_id: str
    parser_name: str
    parser_version = "0.1.0"
    default_query = "midazolam"

    def __init__(
        self,
        *,
        output_root: str = ".",
        allow_network: bool = True,
        timeout_seconds: int = 30,
        run_id: str | None = None,
    ) -> None:
        self.output_root = output_root
        self.allow_network = allow_network
        self.timeout_seconds = timeout_seconds
        self.ingestion_run_id = run_id or f"{utc_now().strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
        self.store = ObjectStore(output_root)
        self.source = self._load_source()
        self.raw_documents: list[RawDocumentMetadata] = []
        self.errors: list[ConnectorError] = []
        self.source_version: str | None = None
        self.used_sample_placeholder = False

    @property
    def connector_name(self) -> str:
        return self.__class__.__name__

    def _load_source(self) -> SourceRegistryEntry:
        for entry in load_source_registry():
            if entry.source_id == self.source_id:
                return entry
        raise ValueError(f"Source registry entry not found for {self.source_id}")

    def run(self, *, query: str | None = None, retmax: int = 3) -> ConnectorRunResult:
        started_at = utc_now()
        try:
            records = self.fetch_and_parse(query=query or self.default_query, retmax=retmax)
            status = "succeeded" if not self.errors else "partially_succeeded"
        except Exception as exc:
            self.errors.append(ConnectorError(stage="fetch_and_parse", message=str(exc), retryable=True))
            records = self.fetch_sample_placeholder(reason=str(exc), query=query or self.default_query)
            self.used_sample_placeholder = True
            status = "partially_succeeded" if records else "failed"

        parsed_records_path = self.store.write_parsed_records(
            source_id=self.source_id,
            ingestion_run_id=self.ingestion_run_id,
            records=records,
        )
        completed_at = utc_now()
        if not self.source_version:
            self.source_version = f"snapshot:{completed_at.strftime('%Y-%m-%d')}:{self.ingestion_run_id}"
        source_checksum = self._source_checksum()
        log = ConnectorRunLog(
            ingestion_run_id=self.ingestion_run_id,
            source_id=self.source_id,
            connector_name=self.connector_name,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            source_version=self.source_version,
            source_checksum=source_checksum,
            raw_document_count=len(self.raw_documents),
            raw_record_count=sum(document.raw_record_count for document in self.raw_documents),
            parsed_record_count=len(records),
            used_sample_placeholder=self.used_sample_placeholder,
            raw_documents=self.raw_documents,
            parsed_records_path=str(parsed_records_path),
            errors=self.errors,
            latest_run_metrics={
                "source_id": self.source_id,
                "status": status,
                "raw_document_count": len(self.raw_documents),
                "raw_record_count": sum(document.raw_record_count for document in self.raw_documents),
                "parsed_record_count": len(records),
                "used_sample_placeholder": self.used_sample_placeholder,
                "source_version": self.source_version,
                "source_checksum": source_checksum,
            },
            source_registry_snapshot=self.source.model_dump(mode="json"),
        )
        ingestion_log_path = self.store.write_ingestion_log(log)
        return ConnectorRunResult(
            ingestion_run_id=self.ingestion_run_id,
            source_id=self.source_id,
            status=status,
            raw_document_count=log.raw_document_count,
            raw_record_count=log.raw_record_count,
            parsed_record_count=log.parsed_record_count,
            used_sample_placeholder=self.used_sample_placeholder,
            parsed_records_path=str(parsed_records_path),
            ingestion_log_path=str(ingestion_log_path),
            raw_documents=self.raw_documents,
            errors=self.errors,
            source_version=self.source_version,
            source_checksum=source_checksum,
        )

    def fetch_and_parse(self, *, query: str, retmax: int) -> list[ParsedSourceRecord]:
        raise NotImplementedError

    def fetch_sample_placeholder(self, *, reason: str, query: str) -> list[ParsedSourceRecord]:
        payload = {
            "source_id": self.source_id,
            "placeholder": True,
            "reason": reason,
            "query": query,
            "message": "Live source retrieval failed or was disabled. This placeholder contains no real evidence values.",
        }
        raw = self.save_raw_json(
            source_url=str(self.source.access_url),
            payload=payload,
            raw_record_count=1,
            is_sample_placeholder=True,
            retrieval_method="sample_placeholder",
            extra_metadata={"reason": reason, "query": query},
        )
        return [
            self.make_record(
                raw=raw,
                source_record_id=f"sample_placeholder:{self.source_id}",
                source_url=str(self.source.access_url),
                source_location="sample_placeholder",
                record_type="sample_placeholder",
                record=payload,
                is_sample_placeholder=True,
            )
        ]

    def _source_checksum(self) -> str | None:
        if not self.raw_documents:
            return None
        digest = sha256()
        for raw_document in sorted(self.raw_documents, key=lambda document: document.raw_storage_path):
            digest.update(raw_document.raw_checksum.encode("utf-8"))
        return digest.hexdigest()

    def fetch_json(self, url: str, *, headers: dict[str, str] | None = None) -> tuple[Any, str | None, int, bytes]:
        payload, content_type, status_code = self.fetch_bytes(url, headers=headers)
        return json.loads(payload.decode("utf-8")), content_type, status_code, payload

    def fetch_bytes(self, url: str, *, headers: dict[str, str] | None = None) -> tuple[bytes, str | None, int]:
        if not self.allow_network:
            raise LiveFetchUnavailable(f"Network fetch disabled for {self.source_id}")
        request_headers = {"User-Agent": "pbpk-stage1-public-kb/0.1 (+public-source-ingestion)"}
        if headers:
            request_headers.update(headers)
        request = Request(url, headers=request_headers)
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = response.read()
                    content_type = response.headers.get("content-type")
                    status_code = getattr(response, "status", None)
                return payload, content_type, int(status_code or 200)
            except HTTPError as exc:
                raise LiveFetchUnavailable(f"HTTP {exc.code} for {url}") from exc
            except (URLError, TimeoutError) as exc:
                last_error = exc
        if isinstance(last_error, TimeoutError):
            raise LiveFetchUnavailable(f"Timeout fetching {url}") from last_error
        if isinstance(last_error, URLError):
            raise LiveFetchUnavailable(f"URL error for {url}: {last_error.reason}") from last_error
        raise LiveFetchUnavailable(f"Fetch failed for {url}")

    def save_raw_bytes(
        self,
        *,
        source_url: str,
        payload: bytes,
        extension: str,
        raw_record_count: int,
        content_type: str | None,
        status_code: int | None,
        source_version: str | None = None,
        is_sample_placeholder: bool = False,
        retrieval_method: str = "http_bytes",
        extra_metadata: dict[str, Any] | None = None,
    ) -> RawDocumentMetadata:
        raw = self.store.write_raw_snapshot(
            source_id=self.source_id,
            ingestion_run_id=self.ingestion_run_id,
            source_url=source_url,
            payload=payload,
            extension=extension,
            content_type=content_type,
            status_code=status_code,
            raw_record_count=raw_record_count,
            source_version=source_version or self.source_version,
            is_sample_placeholder=is_sample_placeholder,
            retrieval_method=retrieval_method,
            extra_metadata=extra_metadata,
        )
        self.raw_documents.append(raw)
        return raw

    def save_raw_json(
        self,
        *,
        source_url: str,
        payload: Any,
        raw_record_count: int,
        is_sample_placeholder: bool = False,
        retrieval_method: str = "http_json",
        status_code: int | None = 200,
        content_type: str | None = "application/json",
        source_version: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        raw_payload_bytes: bytes | None = None,
    ) -> RawDocumentMetadata:
        raw = self.store.write_raw_snapshot(
            source_id=self.source_id,
            ingestion_run_id=self.ingestion_run_id,
            source_url=source_url,
            payload=raw_payload_bytes or json_bytes(payload),
            extension="json",
            content_type=content_type,
            status_code=status_code,
            raw_record_count=raw_record_count,
            source_version=source_version or self.source_version,
            is_sample_placeholder=is_sample_placeholder,
            retrieval_method=retrieval_method,
            extra_metadata=extra_metadata,
        )
        self.raw_documents.append(raw)
        return raw

    def make_record(
        self,
        *,
        raw: RawDocumentMetadata,
        source_record_id: str,
        source_url: str,
        source_location: str,
        record_type: str,
        record: dict[str, Any],
        is_sample_placeholder: bool = False,
    ) -> ParsedSourceRecord:
        return ParsedSourceRecord(
            source_id=self.source_id,
            source_record_id=source_record_id,
            source_url=source_url,
            source_location=source_location,
            raw_document_path=raw.raw_storage_path,
            raw_checksum=raw.raw_checksum,
            download_timestamp=raw.download_timestamp,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            parsed_at=datetime.now(timezone.utc),
            record_type=record_type,
            record=record,
            is_sample_placeholder=is_sample_placeholder,
        )
