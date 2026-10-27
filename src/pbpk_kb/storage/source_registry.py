from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from pbpk_kb.config.sources import SourceRegistryEntry, load_source_registry
from pbpk_kb.storage.db import create_db_engine


def upsert_source_registry_entry(
    connection: Connection,
    entry: SourceRegistryEntry,
    *,
    source_version: str | None = None,
    checksum: str | None = None,
) -> None:
    connection.execute(
        text(
            """
            INSERT INTO source_registry (
                source_id, source_name, source_family, access_url, api_url, access_method,
                expected_format, refresh_frequency, license_notes, connector, parser, entities,
                enabled, source_version, checksum, owner_curator, updated_at
            )
            VALUES (
                :source_id, :source_name, :source_family, :access_url, :api_url, :access_method,
                :expected_format, :refresh_frequency, :license_notes, :connector, :parser, :entities,
                :enabled, :source_version, :checksum, :owner_curator, now()
            )
            ON CONFLICT (source_id) DO UPDATE SET
                source_name = EXCLUDED.source_name,
                source_family = EXCLUDED.source_family,
                access_url = EXCLUDED.access_url,
                api_url = EXCLUDED.api_url,
                access_method = EXCLUDED.access_method,
                expected_format = EXCLUDED.expected_format,
                refresh_frequency = EXCLUDED.refresh_frequency,
                license_notes = EXCLUDED.license_notes,
                connector = EXCLUDED.connector,
                parser = EXCLUDED.parser,
                entities = EXCLUDED.entities,
                enabled = EXCLUDED.enabled,
                source_version = COALESCE(EXCLUDED.source_version, source_registry.source_version),
                checksum = COALESCE(EXCLUDED.checksum, source_registry.checksum),
                owner_curator = COALESCE(EXCLUDED.owner_curator, source_registry.owner_curator),
                updated_at = now()
            """
        ),
        {
            "source_id": entry.source_id,
            "source_name": entry.source_name,
            "source_family": entry.source_family,
            "access_url": str(entry.access_url),
            "api_url": str(entry.api_url) if entry.api_url else None,
            "access_method": entry.access_method,
            "expected_format": entry.expected_format,
            "refresh_frequency": entry.refresh_frequency,
            "license_notes": entry.license_notes,
            "connector": entry.connector,
            "parser": entry.parser,
            "entities": entry.entities,
            "enabled": entry.enabled,
            "source_version": source_version or entry.source_version,
            "checksum": checksum or entry.checksum,
            "owner_curator": entry.owner_curator,
        },
    )


def _load_connector_log(result: dict[str, Any]) -> dict[str, Any]:
    log_path = result.get("ingestion_log_path")
    if not log_path:
        return {}
    path = Path(log_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _error_summary(result: dict[str, Any], log_payload: dict[str, Any]) -> dict[str, Any]:
    errors = result.get("errors") or log_payload.get("errors") or []
    return {
        "used_sample_placeholder": bool(result.get("used_sample_placeholder")),
        "connector_status": result.get("status"),
        "errors": errors,
    }


def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (datetime, date)):
            safe[key] = value.isoformat()
        else:
            safe[key] = value
    return safe


def upsert_ingestion_run_from_connector_result(connection: Connection, result: dict[str, Any]) -> dict[str, Any]:
    log_payload = _load_connector_log(result)
    external_run_id = result.get("ingestion_run_id")
    row = connection.execute(
        text(
            """
            INSERT INTO ingestion_run (
                external_run_id,
                source_id,
                status,
                started_at,
                completed_at,
                source_version,
                raw_record_count,
                parsed_record_count,
                evidence_record_count,
                raw_checksum,
                log_path,
                error_summary
            )
            VALUES (
                :external_run_id,
                :source_id,
                CAST(:status AS ingestion_status),
                CAST(:started_at AS timestamptz),
                CAST(:completed_at AS timestamptz),
                :source_version,
                :raw_record_count,
                :parsed_record_count,
                0,
                :raw_checksum,
                :log_path,
                CAST(:error_summary AS jsonb)
            )
            ON CONFLICT (external_run_id) DO UPDATE SET
                status = EXCLUDED.status,
                completed_at = EXCLUDED.completed_at,
                source_version = EXCLUDED.source_version,
                raw_record_count = EXCLUDED.raw_record_count,
                parsed_record_count = EXCLUDED.parsed_record_count,
                raw_checksum = EXCLUDED.raw_checksum,
                log_path = EXCLUDED.log_path,
                error_summary = EXCLUDED.error_summary
            RETURNING
                ingestion_run_id::text,
                external_run_id,
                source_id,
                status::text,
                started_at,
                completed_at,
                source_version,
                raw_record_count,
                parsed_record_count,
                evidence_record_count,
                raw_checksum,
                log_path,
                error_summary,
                created_at
            """
        ),
        {
            "external_run_id": external_run_id,
            "source_id": result["source_id"],
            "status": result.get("status") or "failed",
            "started_at": log_payload.get("started_at"),
            "completed_at": log_payload.get("completed_at"),
            "source_version": result.get("source_version"),
            "raw_record_count": result.get("raw_record_count") or 0,
            "parsed_record_count": result.get("parsed_record_count") or 0,
            "raw_checksum": result.get("source_checksum"),
            "log_path": result.get("ingestion_log_path"),
            "error_summary": json.dumps(_error_summary(result, log_payload), ensure_ascii=False, sort_keys=True),
        },
    ).mappings().one()
    return _json_safe_row(dict(row))


def upsert_source_registry_entries(
    connection: Connection,
    *,
    source_ids: list[str],
    metadata_by_source: dict[str, dict[str, Any]] | None = None,
) -> None:
    metadata_by_source = metadata_by_source or {}
    entries = [entry for entry in load_source_registry() if entry.source_id in source_ids]
    for entry in entries:
        metadata = metadata_by_source.get(entry.source_id, {})
        upsert_source_registry_entry(
            connection,
            entry,
            source_version=metadata.get("source_version"),
            checksum=metadata.get("source_checksum") or metadata.get("checksum"),
        )


def sync_source_registry_from_connector_results(
    results: list[dict[str, Any]],
    *,
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    metadata_by_source = {
        result["source_id"]: {
            "source_version": result.get("source_version"),
            "source_checksum": result.get("source_checksum"),
        }
        for result in results
        if result.get("source_id")
    }
    if not metadata_by_source:
        return []

    engine = create_db_engine(database_url)
    run_rows: list[dict[str, Any]] = []
    with engine.begin() as connection:
        upsert_source_registry_entries(
            connection,
            source_ids=list(metadata_by_source),
            metadata_by_source=metadata_by_source,
        )
        for result in results:
            if result.get("source_id"):
                run_rows.append(upsert_ingestion_run_from_connector_result(connection, result))
    return [
        {
            "source_id": source_id,
            "source_version": metadata.get("source_version"),
            "source_checksum": metadata.get("source_checksum"),
            "ingestion_run": next((row for row in run_rows if row["source_id"] == source_id), None),
        }
        for source_id, metadata in sorted(metadata_by_source.items())
    ]
