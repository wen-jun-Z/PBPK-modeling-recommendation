from __future__ import annotations

import base64
import struct
from pathlib import Path
from zipfile import ZipFile

import pytest
from defusedxml import ElementTree as SafeET

from pbpk_kb.blueprints.caffeine_dataset import (
    CaffeineDatasetError,
    _UnitTable,
    _extract_pmc535_observations,
    _extract_pmc572_metrics,
    _factor,
    _normalize_unit,
    _read_docx_text,
    _uncertainty_semantics,
    decode_nrbf_single_array,
)


def _nrbf_single_array(values: list[float]) -> str:
    payload = (
        struct.pack("<Biiii", 0, 1, -1, 1, 0)
        + struct.pack("<BiiB", 15, 1, len(values), 11)
        + struct.pack(f"<{len(values)}f", *values)
        + bytes([11])
    )
    return base64.b64encode(payload).decode("ascii")


def _write_dimensions(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<Dimensions>
  <Dimension name="Time" baseUnit="min">
    <Units><Unit name="h" factor="60" /></Units>
  </Dimension>
  <Dimension name="Concentration (mass)" baseUnit="kg/l">
    <Units><Unit name="µg/ml" factor="1E-6" /></Units>
  </Dimension>
</Dimensions>
""",
        encoding="utf-8",
    )


def test_decode_nrbf_single_array_is_strict() -> None:
    encoded = _nrbf_single_array([0.0, 1.25, 2.5])

    assert decode_nrbf_single_array(encoded) == pytest.approx(
        [0.0, 1.25, 2.5]
    )

    payload = bytearray(base64.b64decode(encoded))
    payload[-1] = 0
    with pytest.raises(CaffeineDatasetError, match="MessageEnd"):
        decode_nrbf_single_array(base64.b64encode(payload).decode("ascii"))


def test_unit_factor_parser_rejects_code_and_converts_from_base(
    tmp_path: Path,
) -> None:
    dimensions = tmp_path / "dimensions.xml"
    _write_dimensions(dimensions)
    units = _UnitTable(dimensions)

    assert _factor("1E-6") == pytest.approx(1e-6)
    assert _factor("1/60^2") == pytest.approx(1 / 3600)
    assert _normalize_unit("μg/mL") == "ug/ml"
    assert _normalize_unit("µmol/L") == "umol/l"
    assert units.from_base("Time", "h", 120.0) == pytest.approx(2.0)
    assert units.from_base(
        "Concentration (mass)", "µg/ml", 3.2e-6
    ) == pytest.approx(3.2)
    with pytest.raises(CaffeineDatasetError, match="Unsafe"):
        _factor("__import__('os').system('false')")


def test_docx_study_lock_accepts_word_cell_boundaries_without_spaces(
    tmp_path: Path,
) -> None:
    document = tmp_path / "supplement.docx"
    text = (
        "Wahlländer 1989IV infusion, 10 min3 mg/kg"
        "Blanchard 1983IV infusion, 30 min5 mg/kg"
        "EudraCT 2011-002291-16PO50 mg"
    )
    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/'
        f'wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>{text}'
        "</w:t></w:r></w:p></w:body></w:document>"
    )
    with ZipFile(document, "w") as archive:
        archive.writestr("word/document.xml", xml)

    assert "EudraCT 2011-002291-16PO50 mg" in _read_docx_text(document)


def test_uncertainty_semantics_requires_arithmetic_standard_deviation() -> None:
    sd_column = SafeET.fromstring(
        '<McDataColumn><DataInfo auxiliaryType="ArithmeticStdDev" />'
        "</McDataColumn>"
    )
    sem_column = SafeET.fromstring(
        '<McDataColumn><DataInfo auxiliaryType="ArithmeticStdErr" />'
        "</McDataColumn>"
    )

    assert _uncertainty_semantics(sd_column) == "sd"
    with pytest.raises(CaffeineDatasetError, match="uncertainty semantics"):
        _uncertainty_semantics(sem_column)


def test_extract_pmc535_external_validation_curve(tmp_path: Path) -> None:
    nxml = tmp_path / "pmc535.nxml"
    rows = "".join(
        f"<tr><td>{time}</td><td>0.0 ± 0.0</td><td>0.0 ± 0.0</td>"
        f"<td>{mean} ± {sd}</td></tr>"
        for time, mean, sd in (
            ("Baseline", "0.00", "0.00"),
            ("30 min", "3.31", "0.90"),
            ("60 min", "3.90", "1.07"),
            ("120 min", "3.11", "0.71"),
            ("180 min", "2.48", "0.64"),
            ("240 min", "2.36", "1.04"),
        )
    )
    nxml.write_text(
        "<article><body><table-wrap><label>Table 2</label>"
        "<caption>Serum caffeine concentration by source</caption>"
        "<table><thead><tr><th>Time</th><th>A</th><th>B</th><th>C</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
        "</table-wrap></body></article>",
        encoding="utf-8",
    )

    observations = _extract_pmc535_observations(
        nxml,
        artifact_id="pmc5350996_nxml",
    )

    assert len(observations) == 6
    assert observations[0].time_min == 0.0
    assert observations[1].source_value == 3.31
    assert observations[1].canonical_value_umol_per_l == pytest.approx(
        3.31 * 1000 / 194.19
    )
    assert observations[-1].source_uncertainty == 1.04


def test_extract_pmc572_secondary_metrics(tmp_path: Path) -> None:
    nxml = tmp_path / "pmc572.nxml"
    rows = "".join(
        f"<tr><td>{name}</td><td>gum</td><td>{beverage}</td></tr>"
        for name, beverage in (
            ("AUC0-t (h·ng/mL)", "8,100 (34.9)"),
            ("AUC0-∞ (h·ng/mL)", "8,520 (35.2)"),
            ("Cmax (ng/mL)", "1,300 (28.6)"),
            ("tmaxa (h)", "0.625 (0.333-1.50)"),
            ("t1/2b (h)", "4.70 (1.26)"),
        )
    )
    nxml.write_text(
        "<article><body><table-wrap><label>Table 2.</label>"
        "<caption>Pharmacokinetic Parameters for caffeine</caption>"
        "<table><thead><tr><th>Parameter</th><th>Gum</th>"
        f"<th>Beverage</th></tr></thead><tbody>{rows}</tbody></table>"
        "</table-wrap></body></article>",
        encoding="utf-8",
    )

    metrics = _extract_pmc572_metrics(
        nxml,
        artifact_id="pmc5724581_nxml",
    )

    assert [item.endpoint for item in metrics] == [
        "auc_0_t",
        "auc_0_inf",
        "cmax",
        "tmax",
        "half_life",
    ]
    assert metrics[0].value == 8100.0
    assert metrics[3].variation == pytest.approx([0.333, 1.5])
