from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class NormalizedUnit:
    raw_unit: str | None
    normalized_unit: str | None
    canonical_unit: str | None
    dimension: str | None
    multiplier_to_canonical: float | None = None


_UNIT_ALIASES: dict[str, NormalizedUnit] = {
    "ratio": NormalizedUnit("ratio", "ratio", "ratio", "dimensionless_ratio", 1.0),
    "fold": NormalizedUnit("fold", "fold", "ratio", "dimensionless_ratio", 1.0),
    "fold change": NormalizedUnit("fold change", "fold", "ratio", "dimensionless_ratio", 1.0),
    "%": NormalizedUnit("%", "%", "%", "fraction_percent", 1.0),
    "percent": NormalizedUnit("percent", "%", "%", "fraction_percent", 1.0),
    "% reduction": NormalizedUnit("% reduction", "% reduction", "%", "relative_change_percent", 1.0),
    "% increase": NormalizedUnit("% increase", "% increase", "%", "relative_change_percent", 1.0),
    "%reduction": NormalizedUnit("% reduction", "% reduction", "%", "relative_change_percent", 1.0),
    "%increase": NormalizedUnit("% increase", "% increase", "%", "relative_change_percent", 1.0),
    "day": NormalizedUnit("day", "d", "h", "time", 24.0),
    "days": NormalizedUnit("days", "d", "h", "time", 24.0),
    "d": NormalizedUnit("d", "d", "h", "time", 24.0),
    "week": NormalizedUnit("week", "week", "h", "time", 168.0),
    "weeks": NormalizedUnit("weeks", "week", "h", "time", 168.0),
    "hour": NormalizedUnit("hour", "h", "h", "time", 1.0),
    "hours": NormalizedUnit("hours", "h", "h", "time", 1.0),
    "hr": NormalizedUnit("hr", "h", "h", "time", 1.0),
    "hrs": NormalizedUnit("hrs", "h", "h", "time", 1.0),
    "h": NormalizedUnit("h", "h", "h", "time", 1.0),
    "min": NormalizedUnit("min", "min", "h", "time", 1.0 / 60.0),
    "minute": NormalizedUnit("minute", "min", "h", "time", 1.0 / 60.0),
    "minutes": NormalizedUnit("minutes", "min", "h", "time", 1.0 / 60.0),
    "g/mol": NormalizedUnit("g/mol", "g/mol", "g/mol", "molar_mass", 1.0),
    "mg": NormalizedUnit("mg", "mg", "mg", "mass", 1.0),
    "ug": NormalizedUnit("ug", "ug", "mg", "mass", 0.001),
    "mcg": NormalizedUnit("mcg", "ug", "mg", "mass", 0.001),
    "kg": NormalizedUnit("kg", "kg", "kg", "body_weight", 1.0),
    "ml": NormalizedUnit("mL", "mL", "L", "volume", 0.001),
    "l": NormalizedUnit("l", "L", "L", "volume", 1.0),
    "L": NormalizedUnit("L", "L", "L", "volume", 1.0),
    "ml/kg": NormalizedUnit("mL/kg", "mL/kg", "L/kg", "volume_per_weight", 0.001),
    "l/kg": NormalizedUnit("l/kg", "L/kg", "L/kg", "volume_per_weight", 1.0),
    "L/kg": NormalizedUnit("L/kg", "L/kg", "L/kg", "volume_per_weight", 1.0),
    "ml/min": NormalizedUnit("mL/min", "mL/min", "mL/min", "clearance", 1.0),
    "ml/min/kg": NormalizedUnit("ml/min/kg", "mL/min/kg", "mL/min/kg", "clearance_per_weight", 1.0),
    "mL/min/kg": NormalizedUnit("mL/min/kg", "mL/min/kg", "mL/min/kg", "clearance_per_weight", 1.0),
    "l/h": NormalizedUnit("L/h", "L/h", "L/h", "clearance", 1.0),
    "l/hr": NormalizedUnit("L/hr", "L/h", "L/h", "clearance", 1.0),
    "l/h/kg": NormalizedUnit("L/h/kg", "L/h/kg", "L/h/kg", "clearance_per_weight", 1.0),
    "ml/h/kg": NormalizedUnit("mL/h/kg", "mL/h/kg", "L/h/kg", "clearance_per_weight", 0.001),
    "ng/ml": NormalizedUnit("ng/ml", "ng/mL", "ng/mL", "concentration", 1.0),
    "ng/mL": NormalizedUnit("ng/mL", "ng/mL", "ng/mL", "concentration", 1.0),
    "ug/ml": NormalizedUnit("ug/mL", "ug/mL", "ng/mL", "concentration", 1000.0),
    "mcg/ml": NormalizedUnit("mcg/mL", "ug/mL", "ng/mL", "concentration", 1000.0),
    "mg/l": NormalizedUnit("mg/L", "mg/L", "ng/mL", "concentration", 1000.0),
    "ng*h/ml": NormalizedUnit("ng*h/mL", "ng*h/mL", "ng*h/mL", "auc", 1.0),
    "ng*hr/ml": NormalizedUnit("ng*hr/mL", "ng*h/mL", "ng*h/mL", "auc", 1.0),
    "ug*h/ml": NormalizedUnit("ug*h/mL", "ug*h/mL", "ng*h/mL", "auc", 1000.0),
    "ug*hr/ml": NormalizedUnit("ug*hr/mL", "ug*h/mL", "ng*h/mL", "auc", 1000.0),
    "mcg*h/ml": NormalizedUnit("mcg*h/mL", "ug*h/mL", "ng*h/mL", "auc", 1000.0),
    "mcg*hr/ml": NormalizedUnit("mcg*hr/mL", "ug*h/mL", "ng*h/mL", "auc", 1000.0),
    "mg*h/l": NormalizedUnit("mg*h/L", "mg*h/L", "ng*h/mL", "auc", 1000.0),
    "mg*hr/l": NormalizedUnit("mg*hr/L", "mg*h/L", "ng*h/mL", "auc", 1000.0),
    "mg/kg": NormalizedUnit("mg/kg", "mg/kg", "mg/kg", "dose_per_bw", 1.0),
    "ug/kg": NormalizedUnit("ug/kg", "ug/kg", "mg/kg", "dose_per_bw", 0.001),
    "mcg/kg": NormalizedUnit("mcg/kg", "ug/kg", "mg/kg", "dose_per_bw", 0.001),
    "mg/kg/h": NormalizedUnit("mg/kg/h", "mg/kg/h", "mg/kg/h", "dose_rate_per_bw", 1.0),
    "mg/kg/hr": NormalizedUnit("mg/kg/hr", "mg/kg/h", "mg/kg/h", "dose_rate_per_bw", 1.0),
    "ug/kg/min": NormalizedUnit("ug/kg/min", "ug/kg/min", "mg/kg/h", "dose_rate_per_bw", 0.06),
    "mcg/kg/min": NormalizedUnit("mcg/kg/min", "ug/kg/min", "mg/kg/h", "dose_rate_per_bw", 0.06),
    "mg/h": NormalizedUnit("mg/h", "mg/h", "mg/h", "dose_rate", 1.0),
    "mg/hr": NormalizedUnit("mg/hr", "mg/h", "mg/h", "dose_rate", 1.0),
    "mg/m2": NormalizedUnit("mg/m2", "mg/m2", "mg/m2", "dose_per_bsa", 1.0),
    "mg/m^2": NormalizedUnit("mg/m^2", "mg/m2", "mg/m2", "dose_per_bsa", 1.0),
    # [FIX 2026-07-02 · claude-opus-4-8] (F4) common PK units that previously fell through to dimension="unknown".
    # Molar concentrations canonicalize to nM. These are EXACT-CASE keys only (matched by the case-sensitive first
    # lookup): lowercase "nm"/"mm"/"um" are intentionally NOT added, so nanometer/millimeter/micrometer stay
    # "unknown" rather than being mis-converted to molar.
    "nM": NormalizedUnit("nM", "nM", "nM", "molar_concentration", 1.0),
    "uM": NormalizedUnit("uM", "uM", "nM", "molar_concentration", 1000.0),
    "µM": NormalizedUnit("µM", "uM", "nM", "molar_concentration", 1000.0),
    "μM": NormalizedUnit("μM", "uM", "nM", "molar_concentration", 1000.0),
    "mM": NormalizedUnit("mM", "mM", "nM", "molar_concentration", 1_000_000.0),
    "mg/day": NormalizedUnit("mg/day", "mg/day", "mg/day", "dose_rate", 1.0),
    "mg/d": NormalizedUnit("mg/d", "mg/day", "mg/day", "dose_rate", 1.0),
    "ml/min/1.73m2": NormalizedUnit("mL/min/1.73m2", "mL/min/1.73m2", "mL/min/1.73m2", "clearance_bsa_normalized", 1.0),
    "l/hr/kg": NormalizedUnit("L/hr/kg", "L/h/kg", "L/h/kg", "clearance_per_weight", 1.0),
    "ntpm": NormalizedUnit("nTPM", "nTPM", "nTPM", "expression", 1.0),
    "nTPM": NormalizedUnit("nTPM", "nTPM", "nTPM", "expression", 1.0),
    "tpm": NormalizedUnit("TPM", "TPM", "TPM", "expression", 1.0),
    "TPM": NormalizedUnit("TPM", "TPM", "TPM", "expression", 1.0),
    "ncpm": NormalizedUnit("nCPM", "nCPM", "nCPM", "expression", 1.0),
    "nCPM": NormalizedUnit("nCPM", "nCPM", "nCPM", "expression", 1.0),
    "intensity": NormalizedUnit("intensity", "intensity", "intensity", "expression", 1.0),
}


def _unit_key(unit: str) -> str:
    key = (
        unit.strip()
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("²", "2")
        .replace("·", "*")
        .replace("•", "*")
        .replace("∙", "*")
        .replace("⋅", "*")
        .replace("∗", "*")
        .replace("×", "*")
        .replace(" ", "")
        .lower()
    )
    # FDA labels also print the AUC multiplication sign as an ASCII dot.
    # Match the complete mass-time/volume form to preserve decimals such as
    # 1.73m2 and never turn concentration per hour into an AUC dimension.
    return re.sub(r"^(ng|ug|mcg|mg)\.(h|hr)/(ml|l)$", r"\1*\2/\3", key)


def normalize_unit(unit: str | None) -> NormalizedUnit:
    if unit is None:
        return NormalizedUnit(None, None, None, None, None)
    cleaned = unit.strip()
    if not cleaned:
        return NormalizedUnit(unit, None, None, None, None)
    return _UNIT_ALIASES.get(cleaned) or _UNIT_ALIASES.get(_unit_key(cleaned)) or NormalizedUnit(
        unit,
        cleaned,
        cleaned,
        "unknown",
        1.0,
    )


def to_canonical_value(value: float | int | None, unit: str | None) -> tuple[float | None, str | None]:
    normalized = normalize_unit(unit)
    if value is None:
        return None, normalized.canonical_unit
    if normalized.multiplier_to_canonical is None:
        return float(value), normalized.canonical_unit
    return float(value) * normalized.multiplier_to_canonical, normalized.canonical_unit
