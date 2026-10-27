from __future__ import annotations

import pytest

from pbpk_kb.recommendation.transferability import (
    assess_numeric_transferability,
)


def test_numeric_transferability_accepts_small_dose_perturbation() -> None:
    result = assess_numeric_transferability(
        requested_value=39.9,
        requested_unit="mg",
        validated_values=[(40.0, "mg")],
    )

    assert result.match_class == "numerically_equivalent"
    assert result.direct_reuse_equivalent is True
    assert result.relative_difference == pytest.approx(0.0025)


def test_numeric_transferability_normalizes_mass_units() -> None:
    result = assess_numeric_transferability(
        requested_value=10_000,
        requested_unit="mcg",
        validated_values=[(10.0, "mg")],
    )

    assert result.match_class == "exact"
    assert result.direct_reuse_equivalent is True
    assert result.canonical_requested_value == pytest.approx(10.0)
    assert result.canonical_unit == "mg"


def test_numeric_transferability_distinguishes_interpolation() -> None:
    result = assess_numeric_transferability(
        requested_value=30.0,
        requested_unit="mg",
        validated_values=[(20.0, "mg"), (40.0, "mg")],
    )

    assert result.match_class == "interpolation"
    assert result.direct_reuse_equivalent is False
    assert result.validated_minimum == pytest.approx(20.0)
    assert result.validated_maximum == pytest.approx(40.0)


def test_numeric_transferability_distinguishes_extrapolation() -> None:
    result = assess_numeric_transferability(
        requested_value=80.0,
        requested_unit="mg",
        validated_values=[(20.0, "mg"), (40.0, "mg")],
    )

    assert result.match_class == "extrapolation"
    assert result.direct_reuse_equivalent is False
    assert result.nearest_value == pytest.approx(40.0)


def test_numeric_transferability_rejects_incompatible_units() -> None:
    result = assess_numeric_transferability(
        requested_value=40.0,
        requested_unit="mg/kg",
        validated_values=[(40.0, "mg")],
    )

    assert result.match_class == "unit_mismatch"
    assert result.direct_reuse_equivalent is False
