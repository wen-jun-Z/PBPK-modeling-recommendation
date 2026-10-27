import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("reviewer_obs", ROOT / "scripts/reviewer_external_observations.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("cell,expected", [("1,234.5 ± 200", 1234.5), ("4.3 (1.5–6.0)", 4.3), ("0.004 (40%)", .004)])
def test_numeric_mean_and_median_cells(cell, expected):
    assert module.number(cell) == expected


@pytest.mark.parametrize("cell", ["—", "<0.1", "0", "NaN", "12 unrelated text"])
def test_unqualified_observation_cells_not_silently_numeric(cell):
    with pytest.raises(ValueError):
        module.number(cell)


def test_blocked_metformin_has_no_extraction_spec():
    with pytest.raises(ValueError, match="No authorized"):
        module.table_spec({"pmcid": "PMC13624280"})
