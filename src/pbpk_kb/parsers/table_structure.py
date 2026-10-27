from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from defusedxml import ElementTree as SafeET

from pbpk_kb.parsers.emitter import clean_text


@dataclass(frozen=True)
class StructuredTable:
    columns: list[str]
    rows: list[list[str]]
    caption: str | None = None
    footnotes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def element_text(element: Any | None) -> str:
    if element is None:
        return ""
    return clean_text(" ".join(element.itertext()))


def descendants(element: Any, name: str) -> list[Any]:
    target = name.lower()
    return [child for child in element.iter() if local_name(child.tag).lower() == target]


def parse_xml_table_fragment(fragment: str) -> list[StructuredTable]:
    sanitized = re.sub(r"<\?xml[^>]*\?>", "", fragment, flags=re.I).strip()
    if not sanitized:
        return []
    try:
        root = SafeET.fromstring(f"<table-root>{sanitized}</table-root>")
    except Exception:
        return []
    tables = [node for node in root.iter() if local_name(node.tag).lower() == "table"]
    return [table for node in tables if (table := parse_xml_table_element(node)) is not None]


def parse_xml_table_element(
    table: Any,
    *,
    caption: str | None = None,
    footnotes: list[str] | None = None,
    preserve_partial_header_rows: bool = False,
) -> StructuredTable | None:
    parent_map = {child: parent for parent in table.iter() for child in list(parent)}
    all_row_nodes = [
        node
        for node in table.iter()
        if local_name(node.tag).lower() == "tr" and _nearest_table(node, parent_map) is table
    ]
    footer_rows = [node for node in all_row_nodes if _has_ancestor(node, parent_map, "tfoot")]
    row_nodes = [node for node in all_row_nodes if node not in footer_rows]
    if not row_nodes:
        return None

    expanded_rows: list[list[str]] = []
    header_flags: list[bool] = []
    spans: dict[int, tuple[int, str]] = {}
    max_columns = 0
    for row_node in row_nodes:
        cells = [
            child
            for child in list(row_node)
            if local_name(child.tag).lower() in {"td", "th"}
        ]
        row: list[str] = []
        column_index = 0

        def consume_pending() -> None:
            nonlocal column_index
            while column_index in spans:
                remaining, value = spans[column_index]
                _set_cell(row, column_index, value)
                if remaining <= 1:
                    spans.pop(column_index, None)
                else:
                    spans[column_index] = (remaining - 1, value)
                column_index += 1

        for cell in cells:
            consume_pending()
            value = element_text(cell)
            colspan = _positive_int(cell.get("colspan"), default=1)
            rowspan = _positive_int(cell.get("rowspan"), default=1)
            for offset in range(colspan):
                target_column = column_index + offset
                _set_cell(row, target_column, value)
                if rowspan > 1:
                    spans[target_column] = (rowspan - 1, value)
            column_index += colspan
        consume_pending()
        if spans:
            for target_column in sorted(index for index in spans if index >= column_index):
                while column_index < target_column:
                    _set_cell(row, column_index, "")
                    column_index += 1
                consume_pending()
        if any(cell for cell in row):
            expanded_rows.append(row)
            header_flags.append(
                bool(cells)
                and all(local_name(cell.tag).lower() == "th" for cell in cells)
                or _has_ancestor(row_node, parent_map, "thead")
            )
            max_columns = max(max_columns, len(row))

    if not expanded_rows:
        return None
    normalized_rows = [row + [""] * (max_columns - len(row)) for row in expanded_rows]
    resolved_caption = clean_text(caption) or element_text(next(iter(descendants(table, "caption")), None)) or None
    if (len(normalized_rows) > 1 and _is_spanning_title_row(normalized_rows[0])
            and (not preserve_partial_header_rows or all(normalized_rows[0]))):
        if resolved_caption is None:
            resolved_caption = next(cell for cell in normalized_rows[0] if cell)
        normalized_rows = normalized_rows[1:]
        header_flags = header_flags[1:]
    header_count = 0
    for is_header in header_flags:
        if not is_header:
            break
        header_count += 1
    if header_count == 0 and len(normalized_rows) > 1 and _looks_like_header(normalized_rows[0]):
        header_count = 1

    if header_count:
        columns = [
            " / ".join(dict.fromkeys(row[index] for row in normalized_rows[:header_count] if row[index]))
            or f"column_{index + 1}"
            for index in range(max_columns)
        ]
        body_rows = normalized_rows[header_count:]
    else:
        columns = [f"column_{index + 1}" for index in range(max_columns)]
        body_rows = normalized_rows

    resolved_footnotes = [clean_text(value) for value in (footnotes or []) if clean_text(value)]
    resolved_footnotes.extend(element_text(node) for node in footer_rows if element_text(node))
    return StructuredTable(
        columns=columns,
        rows=body_rows,
        caption=resolved_caption,
        footnotes=resolved_footnotes,
        metadata={"header_row_count": header_count, "rowspan_colspan_expanded": True},
    )


def parse_html_tables(document: str) -> list[StructuredTable]:
    parser = _HTMLTableCollector()
    parser.feed(document)
    parser.close()
    return parser.tables


def parse_markdown_tables(document: str) -> list[tuple[int, int, StructuredTable]]:
    lines = document.splitlines()
    results: list[tuple[int, int, StructuredTable]] = []
    index = 0
    while index + 1 < len(lines):
        header = lines[index]
        separator = lines[index + 1]
        if "|" not in header or not re.match(r"^\s*\|?\s*:?-{3,}", separator):
            index += 1
            continue
        columns = _split_markdown_row(header)
        if not columns:
            index += 1
            continue
        rows: list[list[str]] = []
        row_index = index + 2
        while row_index < len(lines) and "|" in lines[row_index] and lines[row_index].strip():
            row = _split_markdown_row(lines[row_index])
            rows.append(row + [""] * max(0, len(columns) - len(row)))
            row_index += 1
        if rows:
            results.append(
                (
                    index + 1,
                    row_index,
                    StructuredTable(
                        columns=columns,
                        rows=[row[: len(columns)] for row in rows],
                        metadata={"header_row_count": 1, "markdown_table": True},
                    ),
                )
            )
        index = max(index + 1, row_index)
    return results


def _nearest_table(element: Any, parent_map: dict[Any, Any]) -> Any | None:
    current = parent_map.get(element)
    while current is not None:
        if local_name(current.tag).lower() == "table":
            return current
        current = parent_map.get(current)
    return None


def _has_ancestor(element: Any, parent_map: dict[Any, Any], tag: str) -> bool:
    target = tag.lower()
    current = parent_map.get(element)
    while current is not None:
        if local_name(current.tag).lower() == target:
            return True
        current = parent_map.get(current)
    return False


def _set_cell(row: list[str], index: int, value: str) -> None:
    if len(row) <= index:
        row.extend([""] * (index + 1 - len(row)))
    row[index] = value


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(str(value))
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _looks_like_header(row: list[str]) -> bool:
    populated = [cell for cell in row if cell]
    if not populated:
        return False
    numeric = sum(bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cell.replace(",", ""))) for cell in populated)
    return numeric == 0


def _is_spanning_title_row(row: list[str]) -> bool:
    populated = [cell for cell in row if cell]
    return len(row) > 1 and len(set(populated)) == 1 and len(populated[0]) >= 8


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [clean_text(cell.replace(r"\|", "|")) for cell in re.split(r"(?<!\\)\|", stripped)]


class _HTMLTableCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[StructuredTable] = []
        self._table_depth = 0
        self._rows: list[list[dict[str, Any]]] = []
        self._row: list[dict[str, Any]] | None = None
        self._cell: dict[str, Any] | None = None
        self._caption_parts: list[str] | None = None
        self._footnote_parts: list[str] = []
        self._in_thead = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        attributes = {key.lower(): value for key, value in attrs}
        if lowered == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._rows = []
                self._caption_parts = []
                self._footnote_parts = []
            return
        if self._table_depth != 1:
            return
        if lowered == "thead":
            self._in_thead = True
        elif lowered == "caption":
            self._caption_parts = []
        elif lowered == "tr":
            self._row = []
        elif lowered in {"th", "td"} and self._row is not None:
            self._cell = {
                "text": [],
                "header": lowered == "th" or self._in_thead,
                "rowspan": _positive_int(attributes.get("rowspan"), default=1),
                "colspan": _positive_int(attributes.get("colspan"), default=1),
            }

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "table":
            if self._table_depth == 1:
                built = _build_html_table(self._rows, self._caption_parts or [], self._footnote_parts)
                if built is not None:
                    self.tables.append(built)
            self._table_depth = max(0, self._table_depth - 1)
            return
        if self._table_depth != 1:
            return
        if lowered == "thead":
            self._in_thead = False
        elif lowered in {"th", "td"} and self._cell is not None and self._row is not None:
            self._cell["value"] = clean_text(" ".join(self._cell.pop("text")))
            self._row.append(self._cell)
            self._cell = None
        elif lowered == "tr" and self._row is not None:
            self._rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._table_depth != 1:
            return
        text = clean_text(data)
        if not text:
            return
        if self._cell is not None:
            self._cell["text"].append(text)
        elif self._caption_parts is not None:
            self._caption_parts.append(text)


def _build_html_table(
    source_rows: list[list[dict[str, Any]]],
    caption_parts: list[str],
    footnotes: list[str],
) -> StructuredTable | None:
    if not source_rows:
        return None
    spans: dict[int, tuple[int, str]] = {}
    rows: list[list[str]] = []
    header_flags: list[bool] = []
    max_columns = 0
    for source_row in source_rows:
        row: list[str] = []
        column = 0

        def consume_pending() -> None:
            nonlocal column
            while column in spans:
                remaining, value = spans[column]
                _set_cell(row, column, value)
                if remaining <= 1:
                    spans.pop(column, None)
                else:
                    spans[column] = (remaining - 1, value)
                column += 1

        for cell in source_row:
            consume_pending()
            value = str(cell.get("value") or "")
            colspan = int(cell.get("colspan") or 1)
            rowspan = int(cell.get("rowspan") or 1)
            for offset in range(colspan):
                target = column + offset
                _set_cell(row, target, value)
                if rowspan > 1:
                    spans[target] = (rowspan - 1, value)
            column += colspan
        consume_pending()
        if any(row):
            rows.append(row)
            header_flags.append(bool(source_row) and all(bool(cell.get("header")) for cell in source_row))
            max_columns = max(max_columns, len(row))
    if not rows:
        return None
    rows = [row + [""] * (max_columns - len(row)) for row in rows]
    header_count = 0
    for flag in header_flags:
        if not flag:
            break
        header_count += 1
    if header_count == 0 and len(rows) > 1 and _looks_like_header(rows[0]):
        header_count = 1
    if header_count:
        columns = [
            " / ".join(dict.fromkeys(row[index] for row in rows[:header_count] if row[index]))
            or f"column_{index + 1}"
            for index in range(max_columns)
        ]
        body = rows[header_count:]
    else:
        columns = [f"column_{index + 1}" for index in range(max_columns)]
        body = rows
    return StructuredTable(
        columns=columns,
        rows=body,
        caption=clean_text(" ".join(caption_parts)) or None,
        footnotes=[clean_text(note) for note in footnotes if clean_text(note)],
        metadata={"header_row_count": header_count, "rowspan_colspan_expanded": True},
    )
