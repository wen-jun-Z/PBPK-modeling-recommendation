from __future__ import annotations

from pathlib import Path

import pytest

from pbpk_kb.blueprints.caffeine_runtime_audit import (
    CaffeineRuntimeAuditError,
    TARGET_OUTPUT,
    _target_series,
)


def _write_series(path: Path, values: list[tuple[float, float]]) -> None:
    rows = [
        f'"0","{time}","{concentration}"'
        for time, concentration in values
    ]
    path.write_text(
        (
            f'"IndividualId","Time [min]","{TARGET_OUTPUT} [µmol/l]"\n'
            + "\n".join(rows)
            + "\n"
        ),
        encoding="utf-8",
    )


def test_target_series_requires_finite_increasing_nonzero_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.csv"
    _write_series(path, [(0.0, 0.0), (30.0, 1.25), (60.0, 0.5)])

    summary = _target_series(path)

    assert summary.unit == "µmol/l"
    assert summary.row_count == 3
    assert summary.time_strictly_increasing
    assert summary.all_finite
    assert summary.all_nonnegative
    assert summary.concentration_max == 1.25


def test_target_series_exposes_negative_concentration(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    _write_series(path, [(0.0, 0.0), (30.0, -0.1)])

    summary = _target_series(path)

    assert not summary.all_nonnegative


def test_target_series_rejects_missing_target_output(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text("Time [min],Other\n0,1\n", encoding="utf-8")

    with pytest.raises(CaffeineRuntimeAuditError, match="lacks target"):
        _target_series(path)
