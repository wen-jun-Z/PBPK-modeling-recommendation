from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, time
from hashlib import sha256
from pathlib import Path
from typing import Any, TextIO
from uuid import NAMESPACE_URL, uuid5

from pbpk_kb.parsers.models import ParseAsset
from pbpk_kb.processing.raw_processing_paths import RawProcessingLayout
from pbpk_kb.storage.vector_index import chunk_text


PARSER_VERSION = "raw-parse-v4"


def stable_id(*parts: str) -> str:
    return str(uuid5(NAMESPACE_URL, "|".join(parts)))


def hash_text(text: str) -> str:
    return sha256(text.encode("utf-8", errors="replace")).hexdigest()


def clean_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "binary": True,
            "byte_count": len(value),
            "sha256": sha256(value).hexdigest(),
        }
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


class ParseEmitter:
    OUTPUT_NAMES = (
        "parsed_documents.jsonl",
        "parsed_records.jsonl",
        "parsed_tables.jsonl",
        "text_chunks.jsonl",
        "archive_members.jsonl",
        "asset_links.jsonl",
        "raw_asset_aliases.jsonl",
        "asset_parse_status.jsonl",
        "processing_failures.jsonl",
    )

    def __init__(self, layout: RawProcessingLayout, run_id: str) -> None:
        self.layout = layout
        self.run_id = run_id
        self.handles: dict[str, TextIO] = {}
        self.counts: Counter[str] = Counter()
        self.asset_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.parser_counts: Counter[str] = Counter()
        self.status_counts: Counter[str] = Counter()
        self.failure_counts: Counter[str] = Counter()

    def __enter__(self) -> "ParseEmitter":
        for name in self.OUTPUT_NAMES:
            path = self.layout.path(name, for_write=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handles[name] = path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def write(self, name: str, row: dict[str, Any]) -> None:
        payload = json_safe(row)
        self.handles[name].write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        self.handles[name].write("\n")

    def location(self, asset: ParseAsset, suffix: str) -> str:
        return f"{asset.source_id}:{asset.provenance_name}#{suffix}"

    def provenance_metadata(self, asset: ParseAsset, metadata: dict[str, Any] | None) -> dict[str, Any]:
        result = dict(metadata or {})
        if asset.inner_path:
            result["archive_inner_path"] = asset.inner_path
            result["archive_member_checksum"] = asset.asset_metadata_json.get("archive_member_checksum")
        return result

    def document(
        self,
        asset: ParseAsset,
        *,
        parser_name: str,
        content_kind: str,
        title: str,
        text: str,
        source_location: str,
        section_title: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunkable: bool = True,
    ) -> str | None:
        normalized = clean_text(text)
        if not normalized:
            return None
        parsed_document_id = stable_id(
            "parsed_document",
            asset.raw_asset_id,
            source_location,
            content_kind,
            hash_text(normalized),
        )
        metadata_json = self.provenance_metadata(asset, metadata)
        row = {
            "run_id": self.run_id,
            "parsed_document_id": parsed_document_id,
            "raw_asset_id": asset.raw_asset_id,
            "source_id": asset.source_id,
            "compound": asset.compound,
            "storage_path": asset.storage_path,
            "source_url": asset.source_url,
            "checksum": asset.checksum,
            "parser_name": parser_name,
            "parser_version": PARSER_VERSION,
            "content_kind": content_kind,
            "title": clean_text(title)[:1000],
            "section_title": clean_text(section_title) or None,
            "source_location": source_location,
            "text": normalized,
            "text_hash": hash_text(normalized),
            "metadata_json": metadata_json,
        }
        self.write("parsed_documents.jsonl", row)
        self.counts["parsed_documents"] += 1
        self.asset_counts[asset.raw_asset_id]["parsed_documents"] += 1
        if chunkable:
            self._chunks(asset, parsed_document_id, source_location, normalized, section_title, metadata_json)
        return parsed_document_id

    def _chunks(
        self,
        asset: ParseAsset,
        parsed_document_id: str,
        source_location: str,
        text: str,
        section_title: str | None,
        document_metadata: dict[str, Any],
    ) -> None:
        for index, chunk in enumerate(chunk_text(text, section_title=section_title, max_tokens=180), start=1):
            if len(chunk.text.strip()) < 20:
                continue
            chunk_location = f"{source_location}#chunk={index}"
            row = {
                "run_id": self.run_id,
                "chunk_id": stable_id("chunk", parsed_document_id, str(index), hash_text(chunk.text)),
                "raw_asset_id": asset.raw_asset_id,
                "parsed_document_id": parsed_document_id,
                "parsed_table_id": None,
                "source_id": asset.source_id,
                "compound": asset.compound,
                "storage_path": asset.storage_path,
                "source_url": asset.source_url,
                "checksum": asset.checksum,
                "source_location": chunk_location,
                "chunk_index": index,
                "chunk_text": chunk.text,
                "chunk_text_hash": hash_text(chunk.text),
                "token_count": chunk.token_count,
                "metadata_json": {
                    "section_title": chunk.section_title,
                    "paragraph_number": chunk.paragraph_number,
                    "llm_eligible": bool(document_metadata.get("llm_eligible")),
                    "archive_inner_path": document_metadata.get("archive_inner_path"),
                    "archive_member_checksum": document_metadata.get("archive_member_checksum"),
                },
            }
            self.write("text_chunks.jsonl", row)
            self.counts["text_chunks"] += 1
            self.asset_counts[asset.raw_asset_id]["text_chunks"] += 1

    def record(
        self,
        asset: ParseAsset,
        *,
        parser_name: str,
        record_type: str,
        source_location: str,
        record: Any,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        safe_record = json_safe(record)
        canonical = json.dumps(safe_record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        record_hash = hash_text(canonical)
        parsed_record_id = stable_id("parsed_record", asset.raw_asset_id, source_location, record_type, record_hash)
        row = {
            "run_id": self.run_id,
            "parsed_record_id": parsed_record_id,
            "raw_asset_id": asset.raw_asset_id,
            "source_id": asset.source_id,
            "compound": asset.compound,
            "storage_path": asset.storage_path,
            "source_url": asset.source_url,
            "checksum": asset.checksum,
            "parser_name": parser_name,
            "parser_version": PARSER_VERSION,
            "record_type": record_type,
            "source_location": source_location,
            "record_json": safe_record,
            "record_hash": record_hash,
            "metadata_json": self.provenance_metadata(asset, metadata),
        }
        self.write("parsed_records.jsonl", row)
        self.counts["parsed_records"] += 1
        self.asset_counts[asset.raw_asset_id]["parsed_records"] += 1
        return parsed_record_id

    def table(
        self,
        asset: ParseAsset,
        *,
        parser_name: str,
        source_location: str,
        table_index: int,
        columns: list[Any],
        rows: list[list[Any]],
        sheet_name: str | None = None,
        page_number: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        safe_columns = json_safe(columns)
        safe_rows = json_safe(rows)
        table_text = "\n".join(" | ".join("" if cell is None else str(cell) for cell in row) for row in rows)
        parsed_table_id = stable_id("parsed_table", asset.raw_asset_id, source_location, hash_text(table_text))
        row = {
            "run_id": self.run_id,
            "parsed_table_id": parsed_table_id,
            "raw_asset_id": asset.raw_asset_id,
            "source_id": asset.source_id,
            "compound": asset.compound,
            "storage_path": asset.storage_path,
            "source_url": asset.source_url,
            "checksum": asset.checksum,
            "parser_name": parser_name,
            "parser_version": PARSER_VERSION,
            "source_location": source_location,
            "sheet_name": sheet_name,
            "page_number": page_number,
            "table_index": table_index,
            "row_count": len(rows),
            "column_count": len(columns),
            "columns_json": safe_columns,
            "cells_json": safe_rows,
            "table_text": table_text,
            "table_hash": hash_text(table_text),
            "metadata_json": self.provenance_metadata(asset, metadata),
        }
        self.write("parsed_tables.jsonl", row)
        self.counts["parsed_tables"] += 1
        self.asset_counts[asset.raw_asset_id]["parsed_tables"] += 1
        return parsed_table_id

    def archive_member(self, row: dict[str, Any]) -> None:
        self.write("archive_members.jsonl", row)
        self.counts["archive_members"] += 1

    def asset_link(self, row: dict[str, Any]) -> None:
        self.write("asset_links.jsonl", row)
        self.counts["asset_links"] += 1

    def alias(self, row: dict[str, Any]) -> None:
        self.write("raw_asset_aliases.jsonl", row)
        self.counts["raw_asset_aliases"] += 1

    def status(
        self,
        asset: ParseAsset,
        *,
        parser_name: str,
        status: str,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "run_id": self.run_id,
            "raw_asset_id": asset.raw_asset_id,
            "source_id": asset.source_id,
            "compound": asset.compound,
            "storage_path": asset.storage_path,
            "checksum": asset.checksum,
            "file_family": asset.file_family,
            "parser_name": parser_name,
            "parser_version": PARSER_VERSION,
            "terminal_status": status,
            "output_counts": dict(self.asset_counts[asset.raw_asset_id]),
            "warnings": warnings or [],
            "metadata_json": metadata or {},
        }
        self.write("asset_parse_status.jsonl", row)
        self.status_counts[status] += 1
        self.parser_counts[parser_name] += 1
        self.counts["asset_parse_status"] += 1

    def failure(
        self,
        asset: ParseAsset,
        *,
        parser_name: str,
        failure_type: str,
        error: str,
        source_location: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "run_id": self.run_id,
            "stage": "parse",
            "failure_type": failure_type,
            "source_id": asset.source_id,
            "compound": asset.compound,
            "raw_asset_id": asset.raw_asset_id,
            "storage_path": asset.storage_path,
            "source_url": asset.source_url,
            "checksum": asset.checksum,
            "source_location": source_location,
            "parser_name": parser_name,
            "parser_version": PARSER_VERSION,
            "error": error,
            "payload": payload or {},
        }
        self.write("processing_failures.jsonl", row)
        self.failure_counts[failure_type] += 1
        self.counts["processing_failures"] += 1
