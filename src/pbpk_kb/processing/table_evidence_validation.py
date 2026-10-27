"""Validate a DailyMed table cell against the registered original XML.

Table headings are source evidence, not generated prose. A cached verdict or
candidate metadata alone is never sufficient to bypass numeric ownership.
Unsupported formats keep the ordinary quality checks.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from xml.etree.ElementTree import ParseError

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from pbpk_kb.normalization.unit_normalizer import to_canonical_value
from pbpk_kb.parsers.dailymed_spl import DailyMedSPLParser
from pbpk_kb.parsers.emitter import clean_text
from pbpk_kb.parsers.table_structure import descendants, local_name, parse_xml_table_element
from pbpk_kb.processing.source_context import repair_multilevel_headers

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VERSION = "source-table-cell-v1"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def validate_table_evidence(row):
    if row.get("extraction_method") != "source_structured_table" or row.get("source_id") != "dailymed":
        return None
    path = Path(str(row.get("storage_path") or ""))
    if path.suffix.lower() != ".xml":
        return None
    audit = {"version": VERSION, "accepted": False}
    try:
        from pbpk_kb.processing.table_recovery_stage import (
            DDI_RE, central_estimate_count, detect_parameter, find_unit, is_metabolite_analyte_label,
            parse_table_value, resolve_unique_active_ingredient, table_structure_reliable,
        )
        path = path.resolve()
        require(path.is_relative_to(PROJECT_ROOT.resolve()), "source_outside_project")
        require(path.is_file() and path.stat().st_size <= 50 * 1024**2, "source_missing_or_oversized")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == row.get("checksum"), "source_checksum_mismatch")
        root = SafeET.fromstring(raw)
        target = str(row.get("attributed_compound_name") or row.get("compound_name") or row.get("compound") or "")
        require(bool(target), "target_missing")
        ingredients = DailyMedSPLParser()._extract_active_ingredients(root)
        owner, _ = resolve_unique_active_ingredient([x["name"] for x in ingredients], {target}, "verified")
        require(owner == target, "active_ingredient_not_unique_target")
        location = str(row.get("source_location") or "")
        match = re.fullmatch(re.escape(f"dailymed:{path.name}") + r"#xml_path=//section\[(\d+)\]#table=(\d+)#row=(\d+)", location)
        require(match is not None and all(int(x) > 0 for x in match.groups()), "unsupported_source_location")
        section_no, table_no, row_no = map(int, match.groups())
        section = descendants(root, "section")[section_no - 1]
        text_node = next(x for x in list(section) if local_name(x.tag).lower() == "text")
        table_node = descendants(text_node, "table")[table_no - 1]
        table = parse_xml_table_element(table_node)
        require(table is not None, "table_not_parseable")
        table = repair_multilevel_headers(table)
        require(table_structure_reliable(table.columns, table.rows), "table_structure_unreliable")
        cells = table.rows[row_no - 1]
        columns = table.columns
        quote = " | ".join(clean_text(x) for x in cells)
        require(quote == row.get("source_text") and hashlib.sha256(quote.encode()).hexdigest() == row.get("source_text_hash"),
                "source_row_or_quote_hash_mismatch")
        metadata = row.get("metadata_json") or {}
        stored_columns = metadata.get("table_columns") or []
        # Some historical parsed tables encode an empty heading as "" and
        # others as column_N; neither supplies a scientific label.
        def heading(value, i):
            return "" if value == f"column_{i + 1}" else value
        # The historical parsers used both header conventions. Reconstruct
        # the full multirow heading from the same XML cells when necessary;
        # never add a heading supplied only by candidate metadata.
        if ([heading(c, i) for i, c in enumerate(columns)] !=
                [heading(c, i) for i, c in enumerate(stored_columns)]):
            alternative = repair_multilevel_headers(parse_xml_table_element(table_node, preserve_partial_header_rows=True))
            alt_columns = alternative.columns
            if ([heading(c, i) for i, c in enumerate(alt_columns)] ==
                    [heading(c, i) for i, c in enumerate(stored_columns)]):
                require(alternative.rows[row_no - 1] == cells, "restored_header_changes_selected_row")
                columns = alt_columns
        require(len(stored_columns) == len(columns) and
                [heading(c, i) for i, c in enumerate(columns)] ==
                [heading(c, i) for i, c in enumerate(stored_columns)], "table_headings_mismatch")
        index = metadata.get("value_column_index")
        require(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(cells), "invalid_column_index")
        require(metadata.get("selected_value_location") == f"{location}#col={index + 1}"
                and heading(metadata.get("selected_column_header"), index) == heading(columns[index], index),
                "selected_cell_location_mismatch")
        parameters = {i: detect_parameter(c) for i, c in enumerate(columns) if detect_parameter(c)}
        if parameters:
            header = columns[index]
            require(index in parameters and parameters[index] == row.get("parameter"), "parameter_column_mismatch")
            analyte = " ".join(c for i, c in enumerate(cells) if i not in parameters)
        else:
            labels = [(i, detect_parameter(c)) for i, c in enumerate(cells) if detect_parameter(c)]
            require(len(labels) == 1 and labels[0][0] != index and labels[0][1] == row.get("parameter"), "parameter_row_mismatch")
            header = cells[labels[0][0]]
            analyte = header
        require(header == metadata.get("parameter_header"), "parameter_heading_mismatch")
        require(not DDI_RE.search(" ".join([*columns, *cells])), "interaction_table_requires_separate_review")
        require(metadata.get("analyte_role") == "parent" and not is_metabolite_analyte_label(columns[index])
                and not is_metabolite_analyte_label(analyte), "metabolite_or_unresolved_analyte")
        require(central_estimate_count(cells[index]) == 1, "cell_not_one_estimate")
        parsed = parse_table_value(cells[index], unit_hint=find_unit(header) or find_unit(columns[index]))
        require(parsed is not None, "cell_value_or_unit_unresolved")
        value, _, unit, qualifier = parsed
        require(all(row.get(k) is None for k in ("value_low", "value_high", "canonical_value_low", "canonical_value_high")), "scalar_cell_has_range")
        require(not isinstance(row.get("value"), bool) and row.get("value") is not None, "cell_numeric_value_missing")
        expected, expected_unit = to_canonical_value(value, unit)
        observed, observed_unit = to_canonical_value(row["value"], row.get("unit"))
        require(expected is not None and observed is not None and expected_unit == observed_unit
                and math.isclose(expected, observed, rel_tol=1e-10, abs_tol=1e-12), "cell_value_or_unit_mismatch")
        if row.get("canonical_value") is not None:
            canonical, canonical_unit = to_canonical_value(row["canonical_value"], row.get("canonical_unit"))
            require(canonical is not None and canonical_unit == expected_unit
                    and math.isclose(canonical, expected, rel_tol=1e-10, abs_tol=1e-12), "canonical_value_mismatch")
        require(metadata.get("value_qualifier") == qualifier, "cell_qualifier_mismatch")
        require(row.get("value_qualifier") in (None, qualifier), "cell_qualifier_mismatch")
        audit.update(accepted=True, source_sha256=row["checksum"], source_location=location,
                     selected_value_location=metadata["selected_value_location"], parameter_header=header,
                     selected_cell=cells[index], canonical_value=expected, canonical_unit=expected_unit,
                     value_qualifier=qualifier, target=target)
        return {"accepted": True, "audit": audit}
    except (ValueError, OSError, KeyError, IndexError, TypeError, StopIteration, ParseError, DefusedXmlException) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else "source_table_unreadable_or_invalid"
        audit["reason"] = reason
        return {"accepted": False, "error": "structured_table_" + reason, "audit": audit}
