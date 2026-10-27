from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pbpk_kb.parsers.models import ParseAsset, ParseOutcome


ParserCallable = Callable[[ParseAsset, Any], ParseOutcome]


@dataclass(frozen=True)
class RegisteredParser:
    name: str
    callback: ParserCallable


class ParserRegistry:
    def __init__(self) -> None:
        self._generic: dict[str, RegisteredParser] = {}
        self._source_specific: dict[tuple[str, str], RegisteredParser] = {}

    def register(
        self,
        file_family: str,
        name: str,
        callback: ParserCallable,
        *,
        source_id: str | None = None,
    ) -> None:
        registered = RegisteredParser(name=name, callback=callback)
        if source_id:
            self._source_specific[(source_id, file_family)] = registered
        else:
            self._generic[file_family] = registered

    def resolve(self, asset: ParseAsset) -> RegisteredParser:
        parser = self._source_specific.get((asset.source_id, asset.file_family))
        if parser is None:
            parser = self._generic.get(asset.file_family)
        if parser is None:
            raise KeyError(f"No parser registered for source={asset.source_id} family={asset.file_family}")
        return parser

    def registered_keys(self) -> list[str]:
        keys = [f"*:{family}" for family in self._generic]
        keys.extend(f"{source}:{family}" for source, family in self._source_specific)
        return sorted(keys)
