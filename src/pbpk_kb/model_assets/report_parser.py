from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from pypdf import PdfReader

from pbpk_kb.model_assets.license_policy import sha256_file
from pbpk_kb.model_assets.models import (
    EvaluationDatasetReference,
    EvaluationMetricRecord,
    EvaluationReportSection,
    EvaluationReportSummary,
    EvaluationReportTable,
)


MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\([^)]+\)")
MARKDOWN_TABLE_SEPARATOR = re.compile(
    r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$",
    re.MULTILINE,
)
HTML_TAG = re.compile(r"<[^>]+>")
NUMERIC_VALUE = re.compile(
    r"^\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
    r"\s*(?P<unit>%|[A-Za-zµμ/\-\^0-9]+)?\s*$"
)


def _clean_cell(value: str) -> str:
    value = value.replace("<br />", "; ").replace("<br/>", "; ").replace("<br>", "; ")
    value = HTML_TAG.sub("", value)
    value = value.replace("\\|", "|")
    return re.sub(r"\s+", " ", value).strip()


def _split_markdown_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    return [_clean_cell(cell) for cell in re.split(r"(?<!\\)\|", line)]


def _heading_path(
    headings: list[tuple[int, int, str]],
    line_number: int,
) -> list[str]:
    active: dict[int, str] = {}
    for heading_line, level, heading in headings:
        if heading_line >= line_number:
            break
        active[level] = heading
        for child_level in tuple(active):
            if child_level > level:
                del active[child_level]
    return [active[level] for level in sorted(active)]


def _table_role(heading_path: list[str], headers: list[str], rows: list[list[str]]) -> str:
    context = " ".join([*heading_path, *headers]).casefold()
    if any(term in context for term in ("diagnostic", "goodness", "metric", "gmfe")):
        return "validation_metric"
    if (
        any(term in context for term in ("parameter", "physico", "in vitro", "input"))
        and {"unit", "value"} <= {header.casefold() for header in headers}
    ):
        return "input_parameter"
    if any(term in context for term in ("verification", "validation")):
        return "validation_dataset"
    if "qualification" in context:
        return "qualification_dataset"
    if any(term in context for term in ("model building", "calibration", "parameter identification")):
        return "calibration_dataset"
    flattened = " ".join(" ".join(row) for row in rows[:5]).casefold()
    if "based on model snapshot" in flattened or "osp version" in flattened:
        return "model_metadata"
    if any(header.casefold() in {"publication", "data reference"} for header in headers):
        return "other"
    if "reference" in context:
        return "reference"
    return "other"


def _parse_sections(
    lines: list[str],
) -> tuple[list[EvaluationReportSection], list[tuple[int, int, str]]]:
    raw: list[tuple[int, int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        match = MARKDOWN_HEADING.match(line)
        if not match:
            continue
        marker = re.match(r"^\s{0,3}(#{1,6})\s+", line)
        if marker is None:
            continue
        heading = _clean_cell(match.group(1))
        raw.append((line_number, len(marker.group(1)), heading))
    sections: list[EvaluationReportSection] = []
    for index, (line_number, level, heading) in enumerate(raw):
        end_line = len(lines)
        for next_line, next_level, _next_heading in raw[index + 1 :]:
            if next_level <= level:
                end_line = next_line - 1
                break
        sections.append(
            EvaluationReportSection(
                heading=heading,
                level=level,
                start_line=line_number,
                end_line=max(line_number, end_line),
                heading_path=_heading_path(raw[: index + 1], line_number + 1),
            )
        )
    return sections, raw


def _parse_tables(
    lines: list[str],
    headings: list[tuple[int, int, str]],
    *,
    asset_sha256: str,
) -> list[EvaluationReportTable]:
    output: list[EvaluationReportTable] = []
    index = 0
    while index + 1 < len(lines):
        if "|" not in lines[index] or not MARKDOWN_TABLE_SEPARATOR.match(lines[index + 1]):
            index += 1
            continue
        start = index
        headers = _split_markdown_row(lines[index])
        index += 2
        rows: list[list[str]] = []
        while index < len(lines) and "|" in lines[index] and lines[index].strip():
            row = _split_markdown_row(lines[index])
            if len(row) < len(headers):
                row.extend([""] * (len(headers) - len(row)))
            elif len(row) > len(headers):
                row = row[: len(headers) - 1] + [" | ".join(row[len(headers) - 1 :])]
            rows.append(row)
            index += 1
        path = _heading_path(headings, start + 1)
        table_id = hashlib.sha256(
            f"{asset_sha256}\n{start + 1}\n{headers}".encode()
        ).hexdigest()
        output.append(
            EvaluationReportTable(
                table_id=table_id,
                heading_path=path,
                start_line=start + 1,
                end_line=max(start + 2, index),
                headers=headers,
                rows=rows,
                semantic_role=_table_role(path, headers, rows),
            )
        )
    return output


def _dataset_role(table: EvaluationReportTable, row: list[str]) -> str:
    context = " ".join([*table.heading_path, *row]).casefold()
    if any(term in context for term in ("verification", "validation", "independent")):
        return "validation"
    if "qualification" in context:
        return "qualification"
    if any(
        term in context
        for term in ("model building", "calibration", "parameter identification", "training")
    ):
        return "calibration"
    if table.semantic_role == "validation_dataset":
        return "validation"
    if table.semantic_role == "qualification_dataset":
        return "qualification"
    if table.semantic_role == "calibration_dataset":
        return "calibration"
    return "unresolved"


def _dataset_references(
    tables: list[EvaluationReportTable],
    *,
    source_prefix: str,
) -> list[EvaluationDatasetReference]:
    output: list[EvaluationDatasetReference] = []
    for table in tables:
        normalized_headers = [header.casefold() for header in table.headers]
        if not any(
            term in " ".join(normalized_headers)
            for term in ("publication", "study", "scenario", "data reference", "dataset")
        ):
            continue
        for row_number, row in enumerate(table.rows, start=1):
            non_empty = [cell for cell in row if cell]
            if not non_empty:
                continue
            name = non_empty[0]
            reference = next(
                (
                    cell
                    for header, cell in zip(normalized_headers, row, strict=True)
                    if cell and ("reference" in header or "publication" in header)
                ),
                None,
            )
            description = next(
                (
                    cell
                    for header, cell in zip(normalized_headers, row, strict=True)
                    if cell
                    and any(
                        term in header
                        for term in ("information", "treatment", "description", "arm")
                    )
                ),
                None,
            )
            dataset_id = hashlib.sha256(
                f"{table.table_id}\n{row_number}\n{row}".encode()
            ).hexdigest()
            source_line = table.start_line + 1 + row_number
            output.append(
                EvaluationDatasetReference(
                    dataset_id=dataset_id,
                    role=_dataset_role(table, row),
                    name=name,
                    reference=reference,
                    description=description,
                    table_id=table.table_id,
                    row_number=row_number,
                    source_location=f"{source_prefix}#L{source_line}",
                )
            )
    return output


def _metric_records(
    tables: list[EvaluationReportTable],
    *,
    source_prefix: str,
) -> list[EvaluationMetricRecord]:
    output: list[EvaluationMetricRecord] = []
    for table in tables:
        if table.semantic_role != "validation_metric":
            continue
        for row_number, row in enumerate(table.rows, start=1):
            group = row[0] if row else None
            for column_index, cell in enumerate(row[1:], start=1):
                match = NUMERIC_VALUE.match(cell)
                if match is None:
                    continue
                value = float(match.group("value"))
                if not math.isfinite(value):
                    continue
                metric_name = (
                    table.headers[column_index]
                    if column_index < len(table.headers)
                    else f"column_{column_index + 1}"
                )
                metric_id = hashlib.sha256(
                    f"{table.table_id}\n{row_number}\n{column_index}\n{cell}".encode()
                ).hexdigest()
                source_line = table.start_line + 1 + row_number
                output.append(
                    EvaluationMetricRecord(
                        metric_id=metric_id,
                        metric_name=metric_name,
                        value=value,
                        unit=match.group("unit"),
                        group=group,
                        table_id=table.table_id,
                        row_number=row_number,
                        source_location=f"{source_prefix}#L{source_line}",
                    )
                )
    return output


def parse_markdown_report(
    path: Path,
    *,
    relative_path: str,
    repository_commit: str | None = None,
) -> EvaluationReportSummary:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    asset_sha256 = sha256_file(path)
    sections, raw_headings = _parse_sections(lines)
    headings = [section.heading for section in sections]
    tables = _parse_tables(lines, raw_headings, asset_sha256=asset_sha256)
    source_prefix = (
        f"GITHUB_BLOB:{repository_commit}/{relative_path}"
        if repository_commit
        else f"GITHUB_BLOB:{relative_path}"
    )
    return EvaluationReportSummary(
        relative_path=relative_path,
        asset_sha256=asset_sha256,
        format="markdown",
        title=headings[0] if headings else None,
        headings=headings,
        table_count=len(tables),
        link_count=len(MARKDOWN_LINK.findall(text)),
        text_length=len(text),
        extraction_method="markdown_structural_table_dataset_metric_parser",
        sections=sections,
        tables=tables,
        dataset_references=_dataset_references(
            tables,
            source_prefix=source_prefix,
        ),
        metrics=_metric_records(tables, source_prefix=source_prefix),
    )


def parse_pdf_report(path: Path, *, relative_path: str) -> EvaluationReportSummary:
    warnings: list[str] = []
    reader = PdfReader(str(path), strict=False)
    parts: list[str] = []
    for index, page in enumerate(reader.pages):
        try:
            parts.append(page.extract_text() or "")
        except Exception as exc:  # pypdf exposes backend-specific parse errors.
            warnings.append(f"page_{index + 1}: {type(exc).__name__}: {exc}")
    text = "\n".join(parts)
    title = None
    metadata = reader.metadata
    if metadata and metadata.title:
        title = str(metadata.title).strip() or None
    if title is None:
        title = next((line.strip() for line in text.splitlines() if line.strip()), None)
    return EvaluationReportSummary(
        relative_path=relative_path,
        asset_sha256=sha256_file(path),
        format="pdf",
        title=title,
        headings=[],
        table_count=0,
        link_count=0,
        page_count=len(reader.pages),
        text_length=len(text),
        extraction_method="pypdf_text_extraction",
        warnings=warnings,
    )
