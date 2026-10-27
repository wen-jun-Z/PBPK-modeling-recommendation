from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any


@dataclass
class CSVComparison:
    accepted: bool = True
    row_count: int = 0
    column_count: int = 0
    numeric_cell_count: int = 0
    non_numeric_cell_count: int = 0
    time_axis_cell_count: int = 0
    metadata_row_count: int = 0
    maximum_absolute_difference: float = 0.0
    maximum_relative_difference: float = 0.0
    worst_numeric_cell: dict[str, Any] | None = None
    mismatches: list[dict[str, Any]] = field(default_factory=list)


def _numeric(value: str) -> tuple[bool, float]:
    try:
        return True, float(value)
    except ValueError:
        return False, 0.0


def _relative_difference(a: float, b: float, absolute: float) -> float:
    denominator = max(abs(a), abs(b))
    if denominator == 0:
        return 0.0
    return absolute / denominator


def compare_csv_files(
    *,
    run_a_path: Path,
    run_b_path: Path,
    simulation_id: str,
    csv_ordinal: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> CSVComparison:
    comparison = CSVComparison()
    with (
        run_a_path.open("r", encoding="utf-8-sig", newline="") as handle_a,
        run_b_path.open("r", encoding="utf-8-sig", newline="") as handle_b,
    ):
        rows_a = csv.reader(handle_a)
        rows_b = csv.reader(handle_b)
        header_a = next(rows_a, None)
        header_b = next(rows_b, None)
        # PK-Sim Population.csv starts with #Project and #PK-Sim version
        # records before its actual column header. Compare every metadata cell
        # exactly rather than treating the first comment as a one-column table
        # or silently discarding differing provenance.
        while ((header_a and header_a[0].startswith("#"))
               or (header_b and header_b[0].startswith("#"))):
            if header_a != header_b:
                comparison.accepted = False
                comparison.mismatches.append({
                    "reason_code": "CSV_METADATA_MISMATCH",
                    "row_number": comparison.metadata_row_count + 1,
                    "run_a_metadata": header_a,
                    "run_b_metadata": header_b,
                })
                return comparison
            comparison.metadata_row_count += 1
            header_a = next(rows_a, None)
            header_b = next(rows_b, None)
        if header_a is None or header_b is None:
            comparison.accepted = False
            comparison.mismatches.append(
                {
                    "reason_code": "CSV_EMPTY",
                    "run_a_rows": int(header_a is not None),
                    "run_b_rows": int(header_b is not None),
                }
            )
            return comparison
        comparison.column_count = len(header_a)
        if header_a != header_b:
            comparison.accepted = False
            comparison.mismatches.append(
                {
                    "reason_code": "CSV_HEADER_MISMATCH",
                    "run_a_header": header_a,
                    "run_b_header": header_b,
                }
            )
            return comparison

        time_columns = {
            index
            for index, name in enumerate(header_a)
            if name.strip().casefold().startswith("time")
        }
        row_count_a = 0
        row_count_b = 0
        for row_index, (row_a, row_b) in enumerate(
            zip_longest(rows_a, rows_b),
            start=2 + comparison.metadata_row_count,
        ):
            if row_a is not None:
                row_count_a += 1
            if row_b is not None:
                row_count_b += 1
            if row_a is None or row_b is None:
                continue
            _compare_csv_row(
                comparison=comparison,
                row_a=row_a,
                row_b=row_b,
                header=header_a,
                time_columns=time_columns,
                row_index=row_index,
                simulation_id=simulation_id,
                csv_ordinal=csv_ordinal,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        comparison.row_count = max(row_count_a, row_count_b)
        if row_count_a != row_count_b:
            comparison.accepted = False
            comparison.mismatches.append(
                {
                    "reason_code": "CSV_ROW_COUNT_MISMATCH",
                    "run_a_row_count": row_count_a,
                    "run_b_row_count": row_count_b,
                }
            )
    return comparison


def _compare_csv_row(
    *,
    comparison: CSVComparison,
    row_a: list[str],
    row_b: list[str],
    header: list[str],
    time_columns: set[int],
    row_index: int,
    simulation_id: str,
    csv_ordinal: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if len(row_a) != len(header) or len(row_b) != len(header):
        comparison.accepted = False
        comparison.mismatches.append(
            {
                "reason_code": "CSV_COLUMN_COUNT_MISMATCH",
                "row_number": row_index,
                "expected": len(header),
                "run_a": len(row_a),
                "run_b": len(row_b),
            }
        )
        return
    for column_index, (value_a, value_b) in enumerate(
        zip(row_a, row_b, strict=True)
    ):
        column_number = column_index + 1
        column_name = header[column_index]
        if column_index in time_columns:
            comparison.time_axis_cell_count += 1
            if value_a != value_b:
                comparison.accepted = False
                comparison.mismatches.append(
                    {
                        "reason_code": "TIME_AXIS_MISMATCH",
                        "row_number": row_index,
                        "column_number": column_number,
                        "column_name": column_name,
                        "run_a_value": value_a,
                        "run_b_value": value_b,
                    }
                )
            continue

        is_number_a, number_a = _numeric(value_a)
        is_number_b, number_b = _numeric(value_b)
        if is_number_a != is_number_b:
            comparison.accepted = False
            comparison.mismatches.append(
                {
                    "reason_code": "CELL_TYPE_MISMATCH",
                    "row_number": row_index,
                    "column_number": column_number,
                    "column_name": column_name,
                    "run_a_value": value_a,
                    "run_b_value": value_b,
                }
            )
            continue
        if not is_number_a:
            comparison.non_numeric_cell_count += 1
            if value_a != value_b:
                comparison.accepted = False
                comparison.mismatches.append(
                    {
                        "reason_code": "NON_NUMERIC_CELL_MISMATCH",
                        "row_number": row_index,
                        "column_number": column_number,
                        "column_name": column_name,
                        "run_a_value": value_a,
                        "run_b_value": value_b,
                    }
                )
            continue

        comparison.numeric_cell_count += 1
        finite_a = math.isfinite(number_a)
        finite_b = math.isfinite(number_b)
        if not finite_a or not finite_b:
            same_non_finite = (
                math.isnan(number_a)
                and math.isnan(number_b)
            ) or number_a == number_b
            if not same_non_finite:
                comparison.accepted = False
                comparison.mismatches.append(
                    {
                        "reason_code": "NON_FINITE_CELL_MISMATCH",
                        "row_number": row_index,
                        "column_number": column_number,
                        "column_name": column_name,
                        "run_a_value": value_a,
                        "run_b_value": value_b,
                    }
                )
            continue

        absolute = abs(number_a - number_b)
        relative = _relative_difference(number_a, number_b, absolute)
        worst = {
            "simulation_id": simulation_id,
            "csv_ordinal": csv_ordinal,
            "row_number": row_index,
            "column_number": column_number,
            "column_name": column_name,
            "run_a_value": number_a,
            "run_b_value": number_b,
            "absolute_difference": absolute,
            "relative_difference": relative,
        }
        if absolute > comparison.maximum_absolute_difference:
            comparison.maximum_absolute_difference = absolute
            comparison.worst_numeric_cell = worst
        if relative > comparison.maximum_relative_difference:
            comparison.maximum_relative_difference = relative
            if comparison.worst_numeric_cell is None:
                comparison.worst_numeric_cell = worst
        allowed = absolute_tolerance + relative_tolerance * max(
            abs(number_a),
            abs(number_b),
        )
        if absolute > allowed:
            comparison.accepted = False
            comparison.mismatches.append(
                {
                    "reason_code": "NUMERIC_TOLERANCE_EXCEEDED",
                    **worst,
                    "allowed_difference": allowed,
                }
            )
