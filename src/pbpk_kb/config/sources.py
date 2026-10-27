from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, HttpUrl

from pbpk_kb.config.settings import get_settings
from pbpk_kb.paths import PROJECT_ROOT


class SourceRegistryEntry(BaseModel):
    source_id: str
    source_name: str
    source_family: str
    access_url: HttpUrl
    api_url: HttpUrl | None = None
    access_method: str
    expected_format: list[str] = Field(default_factory=list)
    refresh_frequency: str
    license_notes: str
    connector: str
    parser: str | None = None
    entities: list[str] = Field(default_factory=list)
    enabled: bool = True
    source_version: str | None = None
    checksum: str | None = None
    owner_curator: str | None = None


def _default_registry_path() -> Path:
    configured_path = Path(get_settings().source_registry_path)
    if configured_path.is_absolute():
        return configured_path
    return PROJECT_ROOT / configured_path


def load_source_registry(path: str | Path | None = None) -> list[SourceRegistryEntry]:
    registry_path = Path(path) if path is not None else _default_registry_path()
    with registry_path.open("r", encoding="utf-8") as registry_file:
        raw_entries: Any = yaml.safe_load(registry_file) or []
    if not isinstance(raw_entries, list):
        raise ValueError(f"Source registry must be a YAML list: {registry_path}")
    return [SourceRegistryEntry.model_validate(entry) for entry in raw_entries]
