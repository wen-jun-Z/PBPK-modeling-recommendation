from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ConnectorError(BaseModel):
    stage: str
    message: str
    retryable: bool = True
    source_record_id: str | None = None


class RawDocumentMetadata(BaseModel):
    source_id: str
    source_url: str
    download_timestamp: datetime
    raw_checksum: str
    raw_storage_path: str
    raw_metadata_path: str
    content_type: str | None = None
    status_code: int | None = None
    raw_record_count: int = 1
    source_version: str | None = None
    is_sample_placeholder: bool = False
    retrieval_method: str
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedSourceRecord(BaseModel):
    source_id: str
    source_record_id: str
    source_url: str
    source_location: str
    raw_document_path: str
    raw_checksum: str
    download_timestamp: datetime
    parser_name: str
    parser_version: str
    parsed_at: datetime
    record_type: str
    record: dict[str, Any]
    is_sample_placeholder: bool = False


class ConnectorRunLog(BaseModel):
    ingestion_run_id: str
    source_id: str
    connector_name: str
    status: Literal["succeeded", "failed", "partially_succeeded"]
    started_at: datetime
    completed_at: datetime
    source_version: str | None = None
    source_checksum: str | None = None
    raw_document_count: int
    raw_record_count: int
    parsed_record_count: int
    used_sample_placeholder: bool = False
    raw_documents: list[RawDocumentMetadata] = Field(default_factory=list)
    parsed_records_path: str | None = None
    errors: list[ConnectorError] = Field(default_factory=list)
    latest_run_metrics: dict[str, Any] = Field(default_factory=dict)
    source_registry_snapshot: dict[str, Any] = Field(default_factory=dict)


class ConnectorRunResult(BaseModel):
    ingestion_run_id: str
    source_id: str
    status: str
    raw_document_count: int
    raw_record_count: int
    parsed_record_count: int
    used_sample_placeholder: bool
    source_version: str | None = None
    source_checksum: str | None = None
    parsed_records_path: str | None = None
    ingestion_log_path: str
    raw_documents: list[RawDocumentMetadata] = Field(default_factory=list)
    errors: list[ConnectorError] = Field(default_factory=list)
