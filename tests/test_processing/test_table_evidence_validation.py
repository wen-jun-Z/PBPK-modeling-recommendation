from copy import deepcopy
from hashlib import sha256

import pytest

from pbpk_kb.processing import table_evidence_validation as validation
from pbpk_kb.processing.auto_quality_stage import deterministic_repair, evaluate_evidence


def table_row(tmp_path, monkeypatch, vertical=False):
    monkeypatch.setattr(validation, "PROJECT_ROOT", tmp_path)
    path = tmp_path / "label.xml"
    body = ("<tr><th>Parameter</th><th>Caffeine</th></tr>"
            "<tr><td>Percent bound to human plasma protein</td><td>greater than 90%</td></tr>" if vertical else
            "<tr><th>Dose</th><th>Mean C max (µg/mL)</th><th>Tmax (h)</th></tr>"
            "<tr><td>100 mg</td><td>6.57</td><td>2</td></tr>")
    path.write_text('<document xmlns="urn:hl7-org:v3"><ingredient classCode="ACTIB">'
                    '<ingredientSubstance><name>CAFFEINE</name></ingredientSubstance></ingredient>'
                    '<section><text><table>' + body + '</table></text></section></document>')
    quote = "Percent bound to human plasma protein | greater than 90%" if vertical else "100 mg | 6.57 | 2"
    columns = ["Parameter", "Caffeine"] if vertical else ["Dose", "Mean C max (µg/mL)", "Tmax (h)"]
    location = f"dailymed:{path.name}#xml_path=//section[1]#table=1#row=1"
    return dict(evidence_id="example", compound_id="drug", compound_name="caffeine", compound_role="target_compound",
                evidence_type="adme" if vertical else "clinical_pk", parameter="protein_binding" if vertical else "Cmax",
                value=90 if vertical else 6.57, unit="%" if vertical else "µg/mL", source_id="dailymed", raw_asset_id="raw",
                storage_path=str(path), checksum=sha256(path.read_bytes()).hexdigest(), source_location=location,
                source_text=quote, source_text_hash=sha256(quote.encode()).hexdigest(),
                extraction_method="source_structured_table", extractor_version="table-recovery-v3-analyte-qualifiers",
                metadata_json=dict(table_columns=columns, parameter_header="Percent bound to human plasma protein" if vertical else columns[1],
                                   value_column_index=1, selected_column_header=columns[1], selected_value_location=location+"#col=2",
                                   analyte_role="parent", value_qualifier=">" if vertical else None))


@pytest.mark.parametrize("vertical", [False, True])
def test_source_cell_with_heading_passes_without_changing_scientific_fact(tmp_path, monkeypatch, vertical):
    row = table_row(tmp_path, monkeypatch, vertical)
    original = deepcopy(row)
    repaired, repairs = deterministic_repair(row)
    result = evaluate_evidence(repaired, repairs)
    assert result["quality_status"] == "auto_validated"
    assert result["table_source_validation"]["accepted"]
    assert row == original
    assert repaired["source_text"] == original["source_text"]
    assert repaired["value"] == original["value"]
    if not vertical:
        assert repaired["canonical_value"] == 6570
        assert repaired["canonical_unit"] == "ng/mL"
    else:
        assert repaired["value_qualifier"] == ">"


@pytest.mark.parametrize("field,value", [
    ("value", 2), ("unit", "ng/mL"), ("canonical_value", 999), ("compound_name", "theophylline"),
    ("parameter", "Tmax"), ("checksum", "0" * 64), ("value", True), ("value_low", 2),
])
def test_incorrect_numeric_cell_or_identity_is_blocked(tmp_path, monkeypatch, field, value):
    row = table_row(tmp_path, monkeypatch)
    row[field] = value
    row["metadata_json"]["table_source_validation"] = {"accepted": True}
    result = evaluate_evidence(row, [])
    assert result["quality_status"] == "quarantined"
    assert any(x.startswith("structured_table_") for x in result["fatal_blockers"])


@pytest.mark.parametrize("field,value", [("value_column_index", 2), ("parameter_header", "Tmax (h)"),
                                         ("selected_value_location", "wrong"), ("analyte_role", "metabolite"),
                                         ("table_columns", ["Dose", "Tmax (h)", "Cmax (µg/mL)"])])
def test_candidate_metadata_does_not_substitute_for_source(tmp_path, monkeypatch, field, value):
    row = table_row(tmp_path, monkeypatch)
    row["metadata_json"][field] = value
    assert not validation.validate_table_evidence(row)["accepted"]


def test_inequality_is_not_an_exact_measurement(tmp_path, monkeypatch):
    row = table_row(tmp_path, monkeypatch, vertical=True)
    row["metadata_json"]["value_qualifier"] = None
    assert validation.validate_table_evidence(row)["error"].endswith("cell_qualifier_mismatch")


def test_changed_source_bytes_are_not_accepted_from_cache(tmp_path, monkeypatch):
    row = table_row(tmp_path, monkeypatch)
    assert validation.validate_table_evidence(row)["accepted"]
    from pathlib import Path
    Path(row["storage_path"]).write_text("<document/>")
    assert not validation.validate_table_evidence(row)["accepted"]


def test_population_header_with_empty_first_cell_stays_in_its_column():
    from defusedxml import ElementTree as ET
    from pbpk_kb.parsers.table_structure import parse_xml_table_element
    from pbpk_kb.processing.source_context import repair_multilevel_headers
    node = ET.fromstring('<table><tr><td></td><td>1 month to &lt; 1 year</td></tr>'
                        '<tr><td>Parameter</td><td>1.0 mg/kg</td></tr>'
                        '<tr><td>Tmax (h)</td><td>3</td></tr></table>')
    table = repair_multilevel_headers(parse_xml_table_element(node, preserve_partial_header_rows=True))
    assert table.columns == ["Parameter", "1 month to < 1 year / 1.0 mg/kg"]
    assert table.rows == [["Tmax (h)", "3"]]
