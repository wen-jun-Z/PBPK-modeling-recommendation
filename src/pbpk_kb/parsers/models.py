from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParseAsset:
    run_id: str
    raw_asset_id: str
    source_id: str
    compound: str | None
    group: str | None
    storage_path: str
    source_url: str | None
    downloaded_at: str | None
    checksum: str
    content_type: str | None
    file_extension: str
    file_family: str
    file_size: int
    request_id: str | None
    request_key: str | None
    asset_metadata_json: dict[str, Any]
    read_path: str | None = None
    inner_path: str | None = None

    @classmethod
    def from_inventory(cls, row: dict[str, Any]) -> "ParseAsset":
        return cls(
            run_id=str(row["run_id"]),
            raw_asset_id=str(row["raw_asset_id"]),
            source_id=str(row["source_id"]),
            compound=row.get("compound"),
            group=row.get("group"),
            storage_path=str(row["storage_path"]),
            source_url=row.get("source_url"),
            downloaded_at=row.get("downloaded_at"),
            checksum=str(row.get("checksum") or ""),
            content_type=row.get("content_type"),
            file_extension=str(row.get("file_extension") or ""),
            file_family=str(row["file_family"]),
            file_size=int(row.get("file_size") or 0),
            request_id=row.get("request_id"),
            request_key=row.get("request_key"),
            asset_metadata_json=dict(row.get("asset_metadata_json") or {}),
        )

    @property
    def physical_path(self) -> Path:
        return Path(self.read_path or self.storage_path)

    @property
    def provenance_name(self) -> str:
        if self.inner_path:
            return f"{Path(self.storage_path).name}!{self.inner_path}"
        return Path(self.storage_path).name


@dataclass
class ParseOutcome:
    status: str = "parsed"
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
