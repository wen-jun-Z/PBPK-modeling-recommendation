from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pbpk_kb.normalization.unit_normalizer import normalize_unit, to_canonical_value


@dataclass(frozen=True)
class ExtractedTissueExpression:
    gene_symbol: str
    protein_name: str
    tissue: str
    expression_value: float | None
    expression_value_text: str | None
    unit: str
    canonical_value: float | None
    canonical_unit: str | None
    unit_dimension: str
    assay_method: str
    source_location_suffix: str
    evidence_text: str


def extract_hpa_tissue_expression(record: dict[str, Any], *, limit: int = 20) -> list[ExtractedTissueExpression]:
    gene = str(record.get("Gene") or "").strip()
    protein_name = str(record.get("Gene description") or gene).strip()
    rows: list[ExtractedTissueExpression] = []
    rows.extend(
        _extract_expression_dict(
            record.get("RNA tissue specific nTPM"),
            gene_symbol=gene,
            protein_name=protein_name,
            unit="nTPM",
            assay_method="HPA RNA tissue specific nTPM",
            source_location_prefix="RNA_TISSUE_NTPM",
            limit=limit,
        )
    )
    rows.extend(
        _extract_expression_dict(
            record.get("Protein tissue specific Intensity"),
            gene_symbol=gene,
            protein_name=protein_name,
            unit="intensity",
            assay_method="HPA protein tissue specific intensity",
            source_location_prefix="PROTEIN_TISSUE_INTENSITY",
            limit=limit,
        )
    )
    return rows[:limit]


def _extract_expression_dict(
    values: Any,
    *,
    gene_symbol: str,
    protein_name: str,
    unit: str,
    assay_method: str,
    source_location_prefix: str,
    limit: int,
) -> list[ExtractedTissueExpression]:
    if not isinstance(values, dict):
        return []
    normalized = normalize_unit(unit)
    rows: list[ExtractedTissueExpression] = []
    for tissue, raw_value in list(values.items())[:limit]:
        numeric_value = _as_float(raw_value)
        canonical_value, canonical_unit = to_canonical_value(numeric_value, unit)
        rows.append(
            ExtractedTissueExpression(
                gene_symbol=gene_symbol,
                protein_name=protein_name,
                tissue=str(tissue),
                expression_value=numeric_value,
                expression_value_text=str(raw_value) if raw_value is not None else None,
                unit=unit,
                canonical_value=canonical_value,
                canonical_unit=canonical_unit or normalized.canonical_unit,
                unit_dimension=normalized.dimension,
                assay_method=assay_method,
                source_location_suffix=f"{source_location_prefix}:{tissue}",
                evidence_text=f"{gene_symbol} {assay_method}: {tissue}={raw_value} {unit}",
            )
        )
    return rows


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
