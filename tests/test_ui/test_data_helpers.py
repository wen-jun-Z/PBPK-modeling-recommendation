from decimal import Decimal

from pbpk_kb.ui import data


def test_serialize_value_handles_decimals_and_lists() -> None:
    assert data.serialize_value(Decimal("325.8")) == 325.8
    assert data.serialize_value(Decimal("36.0")) == 36
    assert data.serialize_value(["PubChem", Decimal("1.5")]) == ["PubChem", 1.5]


def test_evidence_value_prefers_numeric_value_with_unit() -> None:
    row = {"numeric_value": 325.8, "value_text": "325.8", "unit": "g/mol"}

    assert data.evidence_value(row) == "325.8 g/mol"


def test_evidence_value_uses_value_text_for_ranges() -> None:
    row = {"numeric_value": None, "value_text": "9.3 to 11.0", "unit": "mL/min/kg"}

    assert data.evidence_value(row) == "9.3 to 11.0 mL/min/kg"


def test_snippet_collapses_whitespace_and_truncates() -> None:
    assert data.snippet("alpha\n\nbeta   gamma", length=20) == "alpha beta gamma"
    assert data.snippet("x" * 30, length=12) == "xxxxxxxxx..."

from pbpk_kb.ui.layout import normalize_table_rows, serialize_table_cell


def test_serialize_table_cell_converts_nested_values_to_stable_json() -> None:
    assert serialize_table_cell({"b": [2, 1], "a": {"value": 3}}) == (
        '{"a": {"value": 3}, "b": [2, 1]}'
    )
    assert serialize_table_cell({"CYP3A4", "CYP2C9"}) == (
        '["CYP2C9", "CYP3A4"]'
    )


def test_normalize_table_rows_preserves_scalar_values() -> None:
    rows = normalize_table_rows(
        [
            {
                "constraint": "route",
                "passed": True,
                "observed": {"required": "oral", "model": ["oral", "iv"]},
            }
        ]
    )

    assert rows == [
        {
            "constraint": "route",
            "passed": True,
            "observed": '{"model": ["oral", "iv"], "required": "oral"}',
        }
    ]


def test_normalize_table_rows_coerces_only_heterogeneous_columns() -> None:
    rows = normalize_table_rows(
        [
            {"field": "value", "value": 29.37, "confidence": 0.9},
            {"field": "unit", "value": "L/h", "confidence": 0.95},
            {"field": "validated", "value": True, "confidence": 1},
        ]
    )

    assert [row["value"] for row in rows] == ["29.37", "L/h", "True"]
    assert [row["confidence"] for row in rows] == [0.9, 0.95, 1]
