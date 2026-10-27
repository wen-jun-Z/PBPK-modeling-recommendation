import pytest

from pbpk_kb.normalization.unit_normalizer import normalize_unit, to_canonical_value


@pytest.mark.parametrize("unit", [
    "µg.hr/mL", "μg.hr/mL", "ug.hr/mL", "mcg.hr/mL", "µg.h/mL", "µg . hr / mL",
    "mcg*hr/mL", "mcg⋅hr/mL", "mcg∙hr/mL", "mcg•hr/mL", "mcg·hr/mL",
])
def test_microgram_auc_product_aliases_preserve_dimension_and_scale(unit):
    normalized = normalize_unit(unit)
    assert normalized.dimension == "auc"
    assert normalized.normalized_unit == "ug*h/mL"
    assert normalized.canonical_unit == "ng*h/mL"
    assert to_canonical_value(12, unit) == (12000, "ng*h/mL")


def test_dot_product_normalization_preserves_body_surface_area_decimal():
    unit = normalize_unit("mL/min/1.73m2")
    assert unit.dimension == "clearance_bsa_normalized"
    assert to_canonical_value(90, "mL/min/1.73m2") == (90, "mL/min/1.73m2")


@pytest.mark.parametrize("unit", ["µg/hr/mL", "µg.hr/kg", "µg.5hr/mL", "µg..hr/mL"])
def test_unsupported_or_differently_dimensioned_units_are_not_auc(unit):
    assert normalize_unit(unit).dimension == "unknown"
    assert to_canonical_value(12, unit) == (12, unit)

