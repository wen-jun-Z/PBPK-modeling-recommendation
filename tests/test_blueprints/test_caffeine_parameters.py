from __future__ import annotations

from pathlib import Path

import pytest

from pbpk_kb.blueprints.caffeine_parameters import (
    CALIBRATION_DATASETS,
    CALIBRATION_OBJECTIVE,
    CaffeineParameterError,
    CaffeineParameterRecord,
    _birkett_rows_from_words,
    _extract_lelo_fu,
    _extract_pubmed_kinetics,
)


def test_extract_pubmed_kinetics_from_locked_abstract(tmp_path: Path) -> None:
    source = tmp_path / "pubmed.xml"
    source.write_text(
        """<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID>8553685</PMID><Article><Abstract><AbstractText>
Typical Km and Vmax for the N3-demethylation of caffeine were determined
to be 500 (range 220-1200) microM, and 250 (range 115-450)
pmol.mg protein-1.min-1 respectively.
</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>""",
        encoding="utf-8",
    )

    values = _extract_pubmed_kinetics(source)

    assert values["km"] == 500.0
    assert values["km_low"] == 220.0
    assert values["km_high"] == 1200.0
    assert values["vmax"] == 250.0
    assert values["vmax_low"] == 115.0
    assert values["vmax_high"] == 450.0


def test_birkett_coordinate_table_extracts_weight_and_clearance() -> None:
    words = []
    for subject, y, weight, clearance in (
        (1, 600.0, "76", "0.64"),
        (2, 625.0, "74", "0.50"),
        (3, 650.0, "55", "0.58"),
        (4, 674.0, "50", "0.55"),
        (5, 699.0, "54", "0.55"),
        (6, 723.0, "67", "1.01"),
    ):
        words.extend(
            [
                (141.0, y, 145.0, y + 5, str(subject), 0, 0, 0),
                (196.0, y - 2, 204.0, y + 3, weight, 0, 0, 0),
                (386.0, y, 401.0, y + 5, clearance, 0, 0, 0),
            ]
        )

    rows = _birkett_rows_from_words(words)

    assert len(rows) == 6
    assert rows[0].weight_kg == 76.0
    assert rows[-1].renal_clearance_ml_per_min == 1.01
    assert sum(
        item.normalized_clearance_ml_per_min_per_kg for item in rows
    ) / 6 == pytest.approx(0.010330512664107847)


def test_birkett_coordinate_table_rejects_ambiguous_cells() -> None:
    words = [
        (141.0, 600.0, 145.0, 605.0, "1", 0, 0, 0),
        (196.0, 600.0, 204.0, 605.0, "76", 0, 0, 0),
        (198.0, 600.0, 205.0, 605.0, "77", 0, 0, 0),
    ]

    with pytest.raises(CaffeineParameterError, match="anchors drifted"):
        _birkett_rows_from_words(words)


def test_estimated_parameter_requires_locked_objective() -> None:
    common = {
        "parameter_id": "test",
        "name": "test",
        "pksim_path": "Compounds|Caffeine|test",
        "origin": "estimated",
        "value": 1.0,
        "unit": "x",
        "lower_bound": 0.0,
        "upper_bound": 2.0,
        "calibrate": True,
        "calibration_dataset_ids": CALIBRATION_DATASETS,
        "objective_function": CALIBRATION_OBJECTIVE,
        "provenance": [
            {
                "artifact_id": "source",
                "source_location": "field",
                "extraction_method": "structured_json",
                "source_value_text": "1",
                "source_value_text_sha256": "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b",
            }
        ],
    }

    assert CaffeineParameterRecord(**common).calibrate
    with pytest.raises(ValueError, match="objective drifted"):
        CaffeineParameterRecord(**{**common, "objective_function": "other"})


def test_extract_lelo_fu_rejects_non_pdf(tmp_path: Path) -> None:
    path = tmp_path / "not.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(Exception):
        _extract_lelo_fu(path)
