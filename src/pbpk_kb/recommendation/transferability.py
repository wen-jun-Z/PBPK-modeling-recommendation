from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Iterable, Literal


NUMERIC_TRANSFERABILITY_VERSION = "pbpk-numeric-transferability-1.0.0"
DEFAULT_RELATIVE_EQUIVALENCE_TOLERANCE = 0.005


NumericMatchClass = Literal[
    "exact",
    "numerically_equivalent",
    "interpolation",
    "extrapolation",
    "unit_mismatch",
    "scope_missing",
]


_MASS_TO_MG = {
    "g": 1000.0,
    "mg": 1.0,
    "mcg": 0.001,
    "ug": 0.001,
}


def _unit(value: str) -> str:
    normalized = re.sub(r"\s+", "", value.casefold()).replace("µ", "u")
    aliases = {
        "microgram": "mcg",
        "micrograms": "mcg",
        "milligram": "mg",
        "milligrams": "mg",
        "gram": "g",
        "grams": "g",
    }
    return aliases.get(normalized, normalized)


def _canonical(value: float, unit: str) -> tuple[float, str] | None:
    normalized = _unit(unit)
    if normalized in _MASS_TO_MG:
        return float(value) * _MASS_TO_MG[normalized], "mg"
    return None


@dataclass(frozen=True, slots=True)
class NumericTransferability:
    match_class: NumericMatchClass
    direct_reuse_equivalent: bool
    requested_value: float
    requested_unit: str
    nearest_value: float | None
    nearest_unit: str | None
    canonical_requested_value: float | None
    canonical_nearest_value: float | None
    canonical_unit: str | None
    absolute_difference: float | None
    relative_difference: float | None
    equivalence_tolerance: float
    validated_minimum: float | None
    validated_maximum: float | None
    method_version: str = NUMERIC_TRANSFERABILITY_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_numeric_transferability(
    *,
    requested_value: float,
    requested_unit: str,
    validated_values: Iterable[tuple[float, str]],
    relative_equivalence_tolerance: float = (
        DEFAULT_RELATIVE_EQUIVALENCE_TOLERANCE
    ),
) -> NumericTransferability:
    """Classify a numeric request without treating floating values as IDs."""

    if relative_equivalence_tolerance < 0:
        raise ValueError("relative_equivalence_tolerance must be non-negative")
    requested = _canonical(requested_value, requested_unit)
    if requested is None:
        return NumericTransferability(
            match_class="unit_mismatch",
            direct_reuse_equivalent=False,
            requested_value=float(requested_value),
            requested_unit=requested_unit,
            nearest_value=None,
            nearest_unit=None,
            canonical_requested_value=None,
            canonical_nearest_value=None,
            canonical_unit=None,
            absolute_difference=None,
            relative_difference=None,
            equivalence_tolerance=relative_equivalence_tolerance,
            validated_minimum=None,
            validated_maximum=None,
        )

    comparable: list[tuple[float, float, str]] = []
    for value, unit in validated_values:
        canonical = _canonical(value, unit)
        if canonical is not None and canonical[1] == requested[1]:
            comparable.append((canonical[0], float(value), unit))
    if not comparable:
        return NumericTransferability(
            match_class="scope_missing",
            direct_reuse_equivalent=False,
            requested_value=float(requested_value),
            requested_unit=requested_unit,
            nearest_value=None,
            nearest_unit=None,
            canonical_requested_value=requested[0],
            canonical_nearest_value=None,
            canonical_unit=requested[1],
            absolute_difference=None,
            relative_difference=None,
            equivalence_tolerance=relative_equivalence_tolerance,
            validated_minimum=None,
            validated_maximum=None,
        )

    nearest = min(comparable, key=lambda item: abs(requested[0] - item[0]))
    absolute_difference = abs(requested[0] - nearest[0])
    scale = max(abs(requested[0]), abs(nearest[0]), 1e-12)
    relative_difference = absolute_difference / scale
    exact = math.isclose(
        requested[0], nearest[0], rel_tol=1e-12, abs_tol=1e-12
    )
    equivalent = exact or relative_difference <= relative_equivalence_tolerance
    minimum = min(item[0] for item in comparable)
    maximum = max(item[0] for item in comparable)
    if exact:
        match_class: NumericMatchClass = "exact"
    elif equivalent:
        match_class = "numerically_equivalent"
    elif minimum < requested[0] < maximum:
        match_class = "interpolation"
    else:
        match_class = "extrapolation"
    return NumericTransferability(
        match_class=match_class,
        direct_reuse_equivalent=equivalent,
        requested_value=float(requested_value),
        requested_unit=requested_unit,
        nearest_value=nearest[1],
        nearest_unit=nearest[2],
        canonical_requested_value=requested[0],
        canonical_nearest_value=nearest[0],
        canonical_unit=requested[1],
        absolute_difference=absolute_difference,
        relative_difference=relative_difference,
        equivalence_tolerance=relative_equivalence_tolerance,
        validated_minimum=minimum,
        validated_maximum=maximum,
    )
