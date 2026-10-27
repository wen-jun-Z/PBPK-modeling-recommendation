from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from zipfile import ZipFile

from defusedxml import ElementTree as SafeET
from pydantic import Field, model_validator

from pbpk_kb.blueprints.schemas import StrictBlueprintModel
from pbpk_kb.model_assets.compression import decompress_osp_content
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file


DATASET_BUILDER_VERSION = "caffeine-dataset-builder-1.2.0"
DATASET_VERSION = "caffeine_dataset_v3"
CAFFEINE_MOLECULAR_WEIGHT_G_PER_MOL = 194.19


class CaffeineDatasetError(RuntimeError):
    pass


class SourceArtifact(StrictBlueprintModel):
    artifact_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(gt=0)
    license: str = Field(min_length=1)
    redistribution: Literal[
        "noncommercial_no_derivatives",
        "noncommercial_attribution",
        "metadata_only",
    ]


class ObservationProvenance(StrictBlueprintModel):
    artifact_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    repository_id: str | None = None
    repository_name: str | None = None
    sqlite_content_id: int | None = None
    decoded_xml_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    time_column_id: str | None = None
    time_values_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    column_id: str | None = None
    values_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    uncertainty_column_id: str | None = None
    uncertainty_values_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    uncertainty_auxiliary_type: str | None = None
    row_index: int = Field(ge=0)


class ObservationPoint(StrictBlueprintModel):
    observation_id: str = Field(pattern=r"^obs_[0-9a-f]{24}$")
    dataset_id: str = Field(min_length=1)
    study_id: str = Field(min_length=1)
    split: Literal[
        "calibration",
        "internal_holdout",
        "external_validation",
    ]
    compound: Literal["caffeine"] = "caffeine"
    analyte: Literal["caffeine"] = "caffeine"
    matrix: Literal["plasma", "serum"]
    route: Literal["oral", "intravenous_infusion"]
    formulation: str = Field(min_length=1)
    dose_value: float = Field(gt=0)
    dose_unit: Literal["mg", "mg/kg"]
    infusion_duration_min: float | None = Field(default=None, gt=0)
    time_min: float = Field(ge=0)
    source_value: float = Field(ge=0)
    source_unit: Literal["ug/ml", "umol/l"]
    canonical_value_umol_per_l: float = Field(ge=0)
    statistic: Literal["arithmetic_mean"]
    uncertainty_type: Literal["sd"] | None = None
    source_uncertainty: float | None = Field(default=None, ge=0)
    canonical_uncertainty_umol_per_l: float | None = Field(
        default=None,
        ge=0,
    )
    subject_count: int = Field(gt=0)
    use_in_objective: bool
    exclusion_reason: str | None = None
    provenance: ObservationProvenance

    @model_validator(mode="after")
    def route_matches_infusion(self) -> "ObservationPoint":
        if self.route == "intravenous_infusion" and self.infusion_duration_min is None:
            raise ValueError("IV infusion observations require infusion duration.")
        if self.route == "oral" and self.infusion_duration_min is not None:
            raise ValueError("Oral observations cannot define infusion duration.")
        if not self.use_in_objective and not self.exclusion_reason:
            raise ValueError("Excluded observations require a reason.")
        return self


class PKMetric(StrictBlueprintModel):
    metric_id: str = Field(pattern=r"^metric_[0-9a-f]{24}$")
    dataset_id: str = Field(min_length=1)
    study_id: str = Field(min_length=1)
    split: Literal["external_validation_secondary"]
    endpoint: Literal["auc_0_t", "auc_0_inf", "cmax", "tmax", "half_life"]
    value: float = Field(ge=0)
    unit: str = Field(min_length=1)
    statistic: Literal["geometric_mean", "median", "arithmetic_mean"]
    variation_type: Literal["cv_percent", "range", "sd"] | None = None
    variation: list[float] = Field(default_factory=list)
    subject_count: int = Field(gt=0)
    provenance: ObservationProvenance


class DatasetDefinition(StrictBlueprintModel):
    dataset_id: str = Field(min_length=1)
    study_id: str = Field(min_length=1)
    split: Literal[
        "calibration",
        "internal_holdout",
        "external_validation",
        "external_validation_secondary",
    ]
    title: str = Field(min_length=1)
    publication_id: str = Field(min_length=1)
    population: str = Field(min_length=1)
    subject_count: int = Field(gt=0)
    route: str = Field(min_length=1)
    regimen: str = Field(min_length=1)
    observation_ids: list[str] = Field(default_factory=list)
    metric_ids: list[str] = Field(default_factory=list)
    discrepancy_codes: list[str] = Field(default_factory=list)


class CaffeineDatasetManifest(StrictBlueprintModel):
    schema_version: Literal["1.2.0"] = "1.2.0"
    builder_version: Literal[DATASET_BUILDER_VERSION] = DATASET_BUILDER_VERSION
    dataset_version: Literal[DATASET_VERSION] = DATASET_VERSION
    supersedes: Literal["caffeine_dataset_v2"] = "caffeine_dataset_v2"
    change_reason: Literal["REPRODUCIBLE_CANONICAL_MANIFEST_CHECKSUM"] = (
        "REPRODUCIBLE_CANONICAL_MANIFEST_CHECKSUM"
    )
    generated_at: datetime
    source_artifacts: list[SourceArtifact] = Field(min_length=6)
    datasets: list[DatasetDefinition] = Field(min_length=5)
    observation_count: int = Field(gt=0)
    metric_count: int = Field(gt=0)
    split_counts: dict[str, int]
    source_discrepancies: list[dict[str, str]] = Field(min_length=2)
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class _UnitTable:
    def __init__(self, dimensions_xml: Path) -> None:
        root = SafeET.parse(dimensions_xml).getroot()
        self._factors: dict[str, dict[str, float]] = {}
        for dimension in root.findall(".//Dimension"):
            name = dimension.attrib["name"]
            base_unit = _normalize_unit(dimension.attrib["baseUnit"])
            units = {base_unit: 1.0}
            for unit in dimension.findall("./Units/Unit"):
                units[_normalize_unit(unit.attrib["name"])] = _factor(
                    unit.attrib["factor"]
                )
            self._factors[name] = units

    def has_unit(self, dimension: str, unit: str | None) -> bool:
        return bool(unit) and _normalize_unit(unit) in self._factors.get(
            dimension,
            {},
        )

    def from_base(self, dimension: str, unit: str, value: float) -> float:
        normalized = _normalize_unit(unit)
        try:
            factor = self._factors[dimension][normalized]
        except KeyError as exc:
            raise CaffeineDatasetError(
                f"Unknown unit {unit!r} for dimension {dimension!r}."
            ) from exc
        return value / factor


def _factor(expression: str) -> float:
    node = ast.parse(expression.replace("^", "**"), mode="eval")

    def evaluate(item: ast.AST) -> float:
        if isinstance(item, ast.Expression):
            return evaluate(item.body)
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return float(item.value)
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub)):
            value = evaluate(item.operand)
            return value if isinstance(item.op, ast.UAdd) else -value
        if isinstance(item, ast.BinOp) and isinstance(
            item.op,
            (ast.Mult, ast.Div, ast.Pow),
        ):
            left = evaluate(item.left)
            right = evaluate(item.right)
            if isinstance(item.op, ast.Mult):
                return left * right
            if isinstance(item.op, ast.Div):
                return left / right
            return left**right
        raise CaffeineDatasetError(f"Unsafe unit factor expression: {expression!r}")

    value = evaluate(node)
    if not math.isfinite(value) or value <= 0:
        raise CaffeineDatasetError(f"Invalid unit factor: {expression!r}")
    return value


def _normalize_unit(value: str) -> str:
    return (
        value.strip()
        .casefold()
        .replace("\u03bc", "u")
        .replace("\u00b5", "u")
    )


def decode_nrbf_single_array(encoded: str) -> list[float]:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise CaffeineDatasetError("Values is not strict base64.") from exc
    if len(raw) < 28:
        raise CaffeineDatasetError("NRBF Single[] payload is truncated.")
    record, root_id, header_id, major, minor = struct.unpack_from("<Biiii", raw, 0)
    array_record, object_id, length, primitive_type = struct.unpack_from(
        "<BiiB",
        raw,
        17,
    )
    if (record, root_id, header_id, major, minor) != (0, 1, -1, 1, 0):
        raise CaffeineDatasetError("Unexpected NRBF stream header.")
    if (array_record, object_id, primitive_type) != (15, 1, 11):
        raise CaffeineDatasetError("Values is not an NRBF ArraySinglePrimitive Single[].")
    if length < 0 or length > 1_000_000:
        raise CaffeineDatasetError("NRBF array length is outside the accepted range.")
    expected_length = 27 + (length * 4) + 1
    if len(raw) != expected_length or raw[-1] != 11:
        raise CaffeineDatasetError("NRBF array byte length or MessageEnd is invalid.")
    values = list(struct.unpack_from(f"<{length}f", raw, 27))
    if not all(math.isfinite(value) for value in values):
        raise CaffeineDatasetError("NRBF observation array contains non-finite values.")
    return values


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _artifact(
    path: Path,
    *,
    root: Path,
    artifact_id: str,
    role: str,
    source_id: str,
    source_url: str,
    license_name: str,
    redistribution: Literal[
        "noncommercial_no_derivatives",
        "noncommercial_attribution",
        "metadata_only",
    ],
) -> SourceArtifact:
    if not path.is_file() or path.stat().st_size <= 0:
        raise CaffeineDatasetError(f"Source artifact is missing: {path}")
    return SourceArtifact(
        artifact_id=artifact_id,
        role=role,
        source_id=source_id,
        source_location=_relative(path, root),
        source_url=source_url,
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        license=license_name,
        redistribution=redistribution,
    )


def _read_docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        try:
            document = archive.read("word/document.xml")
        except KeyError as exc:
            raise CaffeineDatasetError(f"DOCX has no document.xml: {path}") from exc
    root = SafeET.fromstring(document)
    text = " ".join("".join(root.itertext()).split())
    required = (
        ("Wahll\u00e4nder 1989", r"Wahll\u00e4nder\s*1989"),
        ("IV infusion, 10 min", r"IV\s*infusion,?\s*10\s*min"),
        ("3 mg/kg", r"3\s*mg/kg"),
        ("Blanchard 1983", r"Blanchard\s*1983"),
        ("IV infusion, 30 min", r"IV\s*infusion,?\s*30\s*min"),
        ("5 mg/kg", r"5\s*mg/kg"),
        ("EudraCT 2011-002291-16", r"EudraCT\s*2011-002291-16"),
        ("PO 50 mg", r"PO\s*50\s*mg"),
    )
    missing = [label for label, pattern in required if not re.search(pattern, text)]
    if missing:
        raise CaffeineDatasetError(
            "Supplement metadata is missing locked study statements: "
            + ", ".join(missing)
        )
    return text


def _read_only_sqlite(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()))}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        connection.close()
        raise CaffeineDatasetError(f"SQLite integrity check failed: {integrity}")
    return connection


_REPOSITORIES: dict[str, dict[str, Any]] = {
    "Blanchard_1982.Caffeine_iv.0.5mgkg": {
        "dataset_id": "blanchard_1983_iv_5mgkg",
        "study_id": "blanchard_1983_absolute_bioavailability",
        "split": "calibration",
        "route": "intravenous_infusion",
        "formulation": "aqueous_iv_infusion",
        "dose_value": 5.0,
        "dose_unit": "mg/kg",
        "infusion_duration_min": 30.0,
        "matrix": "plasma",
        "subject_count": 10,
        "expected_count": 17,
    },
    "Blanchard_1982.Caffeine_po.0.5mgkg": {
        "dataset_id": "blanchard_1983_oral_5mgkg",
        "study_id": "blanchard_1983_absolute_bioavailability",
        "split": "calibration",
        "route": "oral",
        "formulation": "aqueous_oral_solution",
        "dose_value": 5.0,
        "dose_unit": "mg/kg",
        "infusion_duration_min": None,
        "matrix": "plasma",
        "subject_count": 10,
        "expected_count": 11,
    },
    "Wahll\u00e4nder_1989_IV_3mgkg": {
        "dataset_id": "wahllander_1989_iv_3mgkg",
        "study_id": "wahllander_1989_caffeine_iv",
        "split": "internal_holdout",
        "route": "intravenous_infusion",
        "formulation": "aqueous_iv_infusion",
        "dose_value": 3.0,
        "dose_unit": "mg/kg",
        "infusion_duration_min": 10.0,
        "matrix": "plasma",
        "subject_count": 8,
        "expected_count": 7,
    },
    "50 mg PO mean": {
        "dataset_id": "eudract_2011_002291_16_oral_50mg",
        "study_id": "eudract_2011_002291_16",
        "split": "external_validation",
        "route": "oral",
        "formulation": "gelatin_capsule",
        "dose_value": 50.0,
        "dose_unit": "mg",
        "infusion_duration_min": None,
        "matrix": "plasma",
        "subject_count": 103,
        "expected_count": 9,
    },
}


def _source_unit(column: Any, units: _UnitTable) -> str:
    dimension = column.attrib["dimension"]
    info = column.find("DataInfo")
    display = info.attrib.get("displayUnitName") if info is not None else None
    if units.has_unit(dimension, display):
        return _normalize_unit(str(display))
    source = info.attrib.get("source", "") if info is not None else ""
    match = re.search(r"\[([^\[\]]+)\]\s*$", source)
    if match and units.has_unit(dimension, match.group(1)):
        return _normalize_unit(match.group(1))
    raise CaffeineDatasetError(
        f"No valid display unit for column {column.attrib.get('name')!r}."
    )


def _canonical_concentration(value: float, unit: str) -> float:
    normalized = _normalize_unit(unit)
    if normalized == "umol/l":
        return value
    if normalized == "ug/ml":
        return value * 1000.0 / CAFFEINE_MOLECULAR_WEIGHT_G_PER_MOL
    raise CaffeineDatasetError(f"Unsupported caffeine concentration unit: {unit}")


def _manifest_unit(unit: str) -> Literal["ug/ml", "umol/l"]:
    normalized = _normalize_unit(unit)
    if normalized == "ug/ml":
        return "ug/ml"
    if normalized == "umol/l":
        return "umol/l"
    raise CaffeineDatasetError(f"Unsupported manifest concentration unit: {unit}")


def _uncertainty_semantics(column: Any) -> Literal["sd"]:
    info = column.find("DataInfo")
    auxiliary_type = info.attrib.get("auxiliaryType") if info is not None else None
    if auxiliary_type != "ArithmeticStdDev":
        raise CaffeineDatasetError(
            "Unsupported observation uncertainty semantics: "
            f"{auxiliary_type!r}."
        )
    return "sd"


def _extract_pksim_observations(
    project_path: Path,
    *,
    dimensions_xml: Path,
    artifact_id: str,
) -> list[ObservationPoint]:
    units = _UnitTable(dimensions_xml)
    connection = _read_only_sqlite(project_path)
    observations: list[ObservationPoint] = []
    try:
        rows = connection.execute(
            """
            SELECT d.Id, d.Name, d.ContentId, c.Data
            FROM DATA_REPOSITORIES d
            JOIN CONTENTS c ON c.Id = d.ContentId
            ORDER BY d.Name, d.Id
            """
        ).fetchall()
        selected = {str(row["Name"]): row for row in rows if row["Name"] in _REPOSITORIES}
        missing = sorted(set(_REPOSITORIES) - set(selected))
        if missing:
            raise CaffeineDatasetError(
                "PK-Sim project lacks locked observed repositories: "
                + ", ".join(missing)
            )
        for repository_name, spec in _REPOSITORIES.items():
            row = selected[repository_name]
            decoded = decompress_osp_content(bytes(row["Data"]))
            if decoded.crc_valid is False:
                raise CaffeineDatasetError(
                    f"OSP content CRC failed for {repository_name}."
                )
            xml_sha = hashlib.sha256(decoded.data).hexdigest()
            root = SafeET.fromstring(decoded.data)
            columns_element = root.find("Columns")
            if columns_element is None:
                raise CaffeineDatasetError(
                    f"Observed repository has no Columns: {repository_name}."
                )
            columns = list(columns_element)
            grids = [item for item in columns if item.tag == "McBaseGrid"]
            values = [
                item
                for item in columns
                if item.tag == "McDataColumn"
                and item.find("DataInfo") is not None
                and item.find("DataInfo").attrib.get("origin") == "Observation"
            ]
            errors = [
                item
                for item in columns
                if item.tag == "McDataColumn"
                and item.find("DataInfo") is not None
                and item.find("DataInfo").attrib.get("origin")
                == "ObservationAuxiliary"
            ]
            if len(grids) != 1 or len(values) != 1 or len(errors) > 1:
                raise CaffeineDatasetError(
                    f"Unexpected observed repository shape: {repository_name}."
                )
            grid = grids[0]
            concentration = values[0]
            raw_times = decode_nrbf_single_array(grid.findtext("Values", ""))
            raw_values = decode_nrbf_single_array(
                concentration.findtext("Values", "")
            )
            raw_errors = (
                decode_nrbf_single_array(errors[0].findtext("Values", ""))
                if errors
                else [None] * len(raw_values)
            )
            uncertainty_type = (
                _uncertainty_semantics(errors[0]) if errors else None
            )
            expected = int(spec["expected_count"])
            if not (
                len(raw_times)
                == len(raw_values)
                == len(raw_errors)
                == expected
            ):
                raise CaffeineDatasetError(
                    f"Point count mismatch for {repository_name}: "
                    f"times={len(raw_times)}, values={len(raw_values)}, "
                    f"errors={len(raw_errors)}, expected={expected}."
                )
            time_unit = _source_unit(grid, units)
            concentration_unit = _source_unit(concentration, units)
            error_unit = _source_unit(errors[0], units) if errors else None
            values_payload = base64.b64decode(
                concentration.findtext("Values", ""),
                validate=True,
            )
            time_values_payload = base64.b64decode(
                grid.findtext("Values", ""),
                validate=True,
            )
            uncertainty_values_payload = (
                base64.b64decode(
                    errors[0].findtext("Values", ""),
                    validate=True,
                )
                if errors
                else None
            )
            values_sha = hashlib.sha256(values_payload).hexdigest()
            time_values_sha = hashlib.sha256(time_values_payload).hexdigest()
            uncertainty_values_sha = (
                hashlib.sha256(uncertainty_values_payload).hexdigest()
                if uncertainty_values_payload is not None
                else None
            )
            for index, (raw_time, raw_value, raw_error) in enumerate(
                zip(raw_times, raw_values, raw_errors, strict=True)
            ):
                time_min = units.from_base("Time", "min", raw_time)
                source_value = units.from_base(
                    concentration.attrib["dimension"],
                    concentration_unit,
                    raw_value,
                )
                source_error = (
                    units.from_base(
                        errors[0].attrib["dimension"],
                        str(error_unit),
                        float(raw_error),
                    )
                    if raw_error is not None
                    else None
                )
                source_manifest_unit = _manifest_unit(concentration_unit)
                use_in_objective = not (
                    repository_name == "50 mg PO mean" and index == 0
                )
                observations.append(
                    ObservationPoint(
                        observation_id=_stable_id(
                            "obs",
                            DATASET_VERSION,
                            spec["dataset_id"],
                            index,
                        ),
                        dataset_id=spec["dataset_id"],
                        study_id=spec["study_id"],
                        split=spec["split"],
                        matrix=spec["matrix"],
                        route=spec["route"],
                        formulation=spec["formulation"],
                        dose_value=spec["dose_value"],
                        dose_unit=spec["dose_unit"],
                        infusion_duration_min=spec["infusion_duration_min"],
                        time_min=time_min,
                        source_value=source_value,
                        source_unit=source_manifest_unit,
                        canonical_value_umol_per_l=_canonical_concentration(
                            source_value,
                            concentration_unit,
                        ),
                        statistic="arithmetic_mean",
                        uncertainty_type=uncertainty_type,
                        source_uncertainty=source_error,
                        canonical_uncertainty_umol_per_l=(
                            _canonical_concentration(source_error, str(error_unit))
                            if source_error is not None
                            else None
                        ),
                        subject_count=spec["subject_count"],
                        use_in_objective=use_in_objective,
                        exclusion_reason=(
                            "NONZERO_PREDOSE_GROUP_MEAN"
                            if not use_in_objective
                            else None
                        ),
                        provenance=ObservationProvenance(
                            artifact_id=artifact_id,
                            source_location=(
                                f"DATA_REPOSITORIES.Id={row['Id']};"
                                f"CONTENTS.Id={row['ContentId']};"
                                f"McDataRepository/Columns/{concentration.tag}"
                                f"[@id='{concentration.attrib['id']}']/Values"
                            ),
                            repository_id=str(row["Id"]),
                            repository_name=repository_name,
                            sqlite_content_id=int(row["ContentId"]),
                            decoded_xml_sha256=xml_sha,
                            time_column_id=grid.attrib["id"],
                            time_values_sha256=time_values_sha,
                            column_id=concentration.attrib["id"],
                            values_sha256=values_sha,
                            uncertainty_column_id=(
                                errors[0].attrib["id"] if errors else None
                            ),
                            uncertainty_values_sha256=uncertainty_values_sha,
                            uncertainty_auxiliary_type=(
                                "ArithmeticStdDev" if errors else None
                            ),
                            row_index=index,
                        ),
                    )
                )
    finally:
        connection.close()
    return observations


def _text(element: Any | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _table(root: Any, label: str, caption_fragment: str) -> Any:
    matches = [
        item
        for item in root.findall(".//table-wrap")
        if _text(item.find("label")).casefold() == label.casefold()
        and caption_fragment.casefold() in _text(item.find("caption")).casefold()
    ]
    if len(matches) != 1:
        raise CaffeineDatasetError(
            f"Expected one {label} containing {caption_fragment!r}; found {len(matches)}."
        )
    return matches[0]


_MEAN_SD = re.compile(r"(?P<mean>\d+(?:\.\d+)?)\s*\u00b1\s*(?P<sd>\d+(?:\.\d+)?)")


def _extract_pmc535_observations(
    nxml_path: Path,
    *,
    artifact_id: str,
) -> list[ObservationPoint]:
    root = SafeET.parse(nxml_path).getroot()
    table = _table(root, "Table 2", "Serum caffeine concentration")
    rows = table.findall(".//tr")
    expected_times = {
        "Baseline": 0.0,
        "30 min": 30.0,
        "60 min": 60.0,
        "120 min": 120.0,
        "180 min": 180.0,
        "240 min": 240.0,
    }
    observations: list[ObservationPoint] = []
    for row_index, row in enumerate(rows[1:], start=1):
        cells = [_text(item) for item in list(row)]
        if len(cells) != 4 or cells[0] not in expected_times:
            raise CaffeineDatasetError(
                f"Unexpected PMC5350996 Table 2 row: {cells!r}"
            )
        match = _MEAN_SD.search(cells[3])
        if not match:
            raise CaffeineDatasetError(
                f"Synthetic caffeine mean/SD is missing in row {cells[0]!r}."
            )
        mean = float(match.group("mean"))
        sd = float(match.group("sd"))
        observations.append(
            ObservationPoint(
                observation_id=_stable_id(
                    "obs",
                    DATASET_VERSION,
                    "pmc5350996_synthetic_200mg",
                    row_index,
                ),
                dataset_id="pmc5350996_synthetic_200mg",
                study_id="pmc5350996_synthetic_caffeine",
                split="external_validation",
                matrix="serum",
                route="oral",
                formulation="synthetic_caffeine_liquid",
                dose_value=200.0,
                dose_unit="mg",
                time_min=expected_times[cells[0]],
                source_value=mean,
                source_unit="ug/ml",
                canonical_value_umol_per_l=_canonical_concentration(
                    mean,
                    "ug/ml",
                ),
                statistic="arithmetic_mean",
                uncertainty_type="sd",
                source_uncertainty=sd,
                canonical_uncertainty_umol_per_l=_canonical_concentration(
                    sd,
                    "ug/ml",
                ),
                subject_count=12,
                use_in_objective=True,
                provenance=ObservationProvenance(
                    artifact_id=artifact_id,
                    source_location=(
                        "article/body/table-wrap[label='Table 2']/"
                        f"tbody/tr[{row_index + 1}]/td[4]"
                    ),
                    row_index=row_index,
                ),
            )
        )
    if len(observations) != 6:
        raise CaffeineDatasetError("PMC5350996 Table 2 must yield six points.")
    return observations


def _first_number(value: str) -> float:
    match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        raise CaffeineDatasetError(f"No numeric value in {value!r}.")
    return float(match.group())


def _extract_pmc572_metrics(
    nxml_path: Path,
    *,
    artifact_id: str,
) -> list[PKMetric]:
    root = SafeET.parse(nxml_path).getroot()
    table = _table(root, "Table 2.", "Pharmacokinetic Parameters")
    definitions = {
        "AUC0-t (h\u00b7ng/mL)": ("auc_0_t", "h*ng/ml", "geometric_mean", "cv_percent"),
        "AUC0-\u221e (h\u00b7ng/mL)": ("auc_0_inf", "h*ng/ml", "geometric_mean", "cv_percent"),
        "Cmax (ng/mL)": ("cmax", "ng/ml", "geometric_mean", "cv_percent"),
        "tmaxa (h)": ("tmax", "h", "median", "range"),
        "t1/2b (h)": ("half_life", "h", "arithmetic_mean", "sd"),
    }
    metrics: list[PKMetric] = []
    for row_index, row in enumerate(table.findall(".//tr")[1:], start=1):
        cells = [_text(item) for item in list(row)]
        if not cells or cells[0] not in definitions:
            continue
        endpoint, unit, statistic, variation_type = definitions[cells[0]]
        raw = cells[2]
        normalized_raw = raw.replace(",", "")
        numbers = [
            float(item)
            for item in re.findall(r"\d+(?:\.\d+)?", normalized_raw)
        ]
        if not numbers:
            raise CaffeineDatasetError(f"No metric values in PMC572 row {cells!r}.")
        metrics.append(
            PKMetric(
                metric_id=_stable_id(
                    "metric",
                    DATASET_VERSION,
                    "pmc5724581_beverage_50_65mg",
                    endpoint,
                ),
                dataset_id="pmc5724581_beverage_50_65mg",
                study_id="pmc5724581_caffeinated_beverage",
                split="external_validation_secondary",
                endpoint=endpoint,
                value=numbers[0],
                unit=unit,
                statistic=statistic,
                variation_type=variation_type,
                variation=numbers[1:],
                subject_count=16,
                provenance=ObservationProvenance(
                    artifact_id=artifact_id,
                    source_location=(
                        "article/body/table-wrap[label='Table 2.']/"
                        f"tbody/tr[{row_index + 1}]/td[3]"
                    ),
                    row_index=row_index,
                ),
            )
        )
    expected = {"auc_0_t", "auc_0_inf", "cmax", "tmax", "half_life"}
    if {item.endpoint for item in metrics} != expected:
        raise CaffeineDatasetError("PMC5724581 PK metric set is incomplete.")
    return metrics


def _write_jsonl_atomic(path: Path, values: list[StrictBlueprintModel]) -> str:
    payload = b"".join(
        json.dumps(
            item.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for item in values
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def build_caffeine_dataset(
    *,
    source_root: Path,
    pmc830_oa_archive: Path,
    pmc830_nxml: Path,
    pmc830_s001_zip: Path,
    pmc830_pksim_project: Path,
    pmc830_s002_docx: Path,
    pmc535_nxml: Path,
    pmc572_nxml: Path,
    dimensions_xml: Path,
    output_dir: Path,
) -> CaffeineDatasetManifest:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CaffeineDatasetError(
            f"Immutable dataset output already contains files: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = source_root.resolve()

    _read_docx_text(pmc830_s002_docx)
    artifacts = [
        _artifact(
            pmc830_oa_archive,
            root=source_root,
            artifact_id="pmc8302243_oa_archive",
            role="pmc_oa_package",
            source_id="pmc_oa",
            source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8302243/",
            license_name="CC BY-NC-ND 4.0",
            redistribution="noncommercial_no_derivatives",
        ),
        _artifact(
            pmc830_nxml,
            root=source_root,
            artifact_id="pmc8302243_nxml",
            role="article_fulltext_xml",
            source_id="pmc_oa",
            source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8302243/",
            license_name="CC BY-NC-ND 4.0",
            redistribution="noncommercial_no_derivatives",
        ),
        _artifact(
            pmc830_s001_zip,
            root=source_root,
            artifact_id="pmc8302243_supinfo_s1_zip",
            role="supplement_model_archive",
            source_id="pmc_oa",
            source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8302243/",
            license_name="CC BY-NC-ND 4.0",
            redistribution="noncommercial_no_derivatives",
        ),
        _artifact(
            pmc830_pksim_project,
            root=source_root,
            artifact_id="pmc8302243_caffeine_pksim5",
            role="supplement_observed_data_container",
            source_id="pmc_oa",
            source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8302243/",
            license_name="CC BY-NC-ND 4.0",
            redistribution="noncommercial_no_derivatives",
        ),
        _artifact(
            pmc830_s002_docx,
            root=source_root,
            artifact_id="pmc8302243_supinfo_s2_docx",
            role="supplement_study_metadata",
            source_id="pmc_oa",
            source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8302243/",
            license_name="CC BY-NC-ND 4.0",
            redistribution="noncommercial_no_derivatives",
        ),
        _artifact(
            pmc535_nxml,
            root=source_root,
            artifact_id="pmc5350996_nxml",
            role="external_validation_fulltext_xml",
            source_id="pmc_oa",
            source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC5350996/",
            license_name="CC BY-NC-ND 4.0",
            redistribution="noncommercial_no_derivatives",
        ),
        _artifact(
            pmc572_nxml,
            root=source_root,
            artifact_id="pmc5724581_nxml",
            role="secondary_validation_fulltext_xml",
            source_id="pmc_oa",
            source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC5724581/",
            license_name="CC BY-NC 4.0",
            redistribution="noncommercial_attribution",
        ),
        _artifact(
            dimensions_xml,
            root=source_root,
            artifact_id="ospsuite_dimensions_v12_4_3",
            role="unit_conversion_definition",
            source_id="osp_suite",
            source_url="https://github.com/Open-Systems-Pharmacology/OSPSuite.Core",
            license_name="GPL-2.0-or-later WITH OSP-Suite-Addendum",
            redistribution="metadata_only",
        ),
    ]

    observations = _extract_pksim_observations(
        pmc830_pksim_project,
        dimensions_xml=dimensions_xml,
        artifact_id="pmc8302243_caffeine_pksim5",
    )
    observations.extend(
        _extract_pmc535_observations(
            pmc535_nxml,
            artifact_id="pmc5350996_nxml",
        )
    )
    metrics = _extract_pmc572_metrics(
        pmc572_nxml,
        artifact_id="pmc5724581_nxml",
    )

    dataset_specs = [
        DatasetDefinition(
            dataset_id="blanchard_1983_iv_5mgkg",
            study_id="blanchard_1983_absolute_bioavailability",
            split="calibration",
            title="The absolute bioavailability of caffeine in man - IV arm",
            publication_id="PMID:6832208",
            population="10 healthy adult males; mean age 21.8 years",
            subject_count=10,
            route="intravenous infusion over 30 min",
            regimen="single 5 mg/kg",
            observation_ids=[
                item.observation_id
                for item in observations
                if item.dataset_id == "blanchard_1983_iv_5mgkg"
            ],
            discrepancy_codes=[
                "REPOSITORY_LABEL_DOSE_CONFLICT_RESOLVED_BY_SUPPLEMENT_AND_PRIMARY_PUBLICATION"
            ],
        ),
        DatasetDefinition(
            dataset_id="blanchard_1983_oral_5mgkg",
            study_id="blanchard_1983_absolute_bioavailability",
            split="calibration",
            title="The absolute bioavailability of caffeine in man - oral arm",
            publication_id="PMID:6832208",
            population="10 healthy adult males; mean age 21.8 years",
            subject_count=10,
            route="aqueous oral solution",
            regimen="single 5 mg/kg",
            observation_ids=[
                item.observation_id
                for item in observations
                if item.dataset_id == "blanchard_1983_oral_5mgkg"
            ],
            discrepancy_codes=[
                "REPOSITORY_LABEL_DOSE_CONFLICT_RESOLVED_BY_SUPPLEMENT_AND_PRIMARY_PUBLICATION"
            ],
        ),
        DatasetDefinition(
            dataset_id="wahllander_1989_iv_3mgkg",
            study_id="wahllander_1989_caffeine_iv",
            split="internal_holdout",
            title=(
                "Effect of ketoconazole and terbinafine on the "
                "pharmacokinetics of caffeine in healthy volunteers"
            ),
            publication_id="Wahllander 1989",
            population="8 healthy adult males; mean age 26 years; mean weight 74 kg",
            subject_count=8,
            route="intravenous infusion over 10 min",
            regimen="single 3 mg/kg",
            observation_ids=[
                item.observation_id
                for item in observations
                if item.dataset_id == "wahllander_1989_iv_3mgkg"
            ],
            discrepancy_codes=["SUPPLEMENT_POINT_COUNT_MISMATCH_10_VS_7"],
        ),
        DatasetDefinition(
            dataset_id="eudract_2011_002291_16_oral_50mg",
            study_id="eudract_2011_002291_16",
            split="external_validation",
            title="Data-driven personalization of a PBPK model for caffeine",
            publication_id="PMID:34053199;NCT01788254",
            population="103 healthy adult volunteers; group mean",
            subject_count=103,
            route="oral gelatin capsule",
            regimen="single 50 mg trimethyl-13C3 caffeine",
            observation_ids=[
                item.observation_id
                for item in observations
                if item.dataset_id == "eudract_2011_002291_16_oral_50mg"
            ],
        ),
        DatasetDefinition(
            dataset_id="pmc5350996_synthetic_200mg",
            study_id="pmc5350996_synthetic_caffeine",
            split="external_validation",
            title="Safety and pharmacokinetics of natural caffeine sources",
            publication_id="PMID:27320048;PMC5350996",
            population="12 healthy nonsmoking adult males; mean age 24.33 years",
            subject_count=12,
            route="oral liquid consumed within 5 min",
            regimen="single 200 mg synthetic caffeine",
            observation_ids=[
                item.observation_id
                for item in observations
                if item.dataset_id == "pmc5350996_synthetic_200mg"
            ],
        ),
        DatasetDefinition(
            dataset_id="pmc5724581_beverage_50_65mg",
            study_id="pmc5724581_caffeinated_beverage",
            split="external_validation_secondary",
            title="Caffeinated chewing gum versus caffeinated beverage",
            publication_id="PMID:29230348;PMC5724581",
            population="16 healthy adults; 8 male and 8 female; mean age 35 years",
            subject_count=16,
            route="oral caffeinated beverage",
            regimen="single actual mean dose 50.65 mg",
            metric_ids=[item.metric_id for item in metrics],
        ),
    ]
    split_studies: dict[str, set[str]] = {}
    for item in dataset_specs:
        split_studies.setdefault(item.split, set()).add(item.study_id)
    primary_splits = (
        split_studies.get("calibration", set()),
        split_studies.get("internal_holdout", set()),
        split_studies.get("external_validation", set()),
    )
    no_study_leakage = all(
        not primary_splits[left] & primary_splits[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    checks = {
        "source_artifacts_nonempty_and_checksummed": all(
            item.byte_size > 0 and len(item.sha256) == 64 for item in artifacts
        ),
        "study_level_split_has_no_leakage": no_study_leakage,
        "calibration_contains_iv_and_oral": {
            item.route
            for item in observations
            if item.split == "calibration"
        }
        == {"intravenous_infusion", "oral"},
        "internal_holdout_is_distinct_iv_study": any(
            item.split == "internal_holdout"
            and item.route == "intravenous_infusion"
            for item in observations
        ),
        "external_validation_has_two_oral_studies": len(
            {
                item.study_id
                for item in observations
                if item.split == "external_validation" and item.route == "oral"
            }
        )
        >= 2,
        "all_observations_finite_nonnegative": all(
            math.isfinite(item.time_min)
            and math.isfinite(item.canonical_value_umol_per_l)
            and item.time_min >= 0
            and item.canonical_value_umol_per_l >= 0
            for item in observations
        ),
        "predose_eudract_retained_but_excluded": sum(
            1
            for item in observations
            if item.study_id == "eudract_2011_002291_16"
            and not item.use_in_objective
            and item.time_min == 0
        )
        == 1,
        "pksim_nrbf_columns_fully_checksummed": all(
            item.provenance.time_column_id
            and item.provenance.time_values_sha256
            and item.provenance.column_id
            and item.provenance.values_sha256
            and (
                item.source_uncertainty is None
                or (
                    item.provenance.uncertainty_column_id
                    and item.provenance.uncertainty_values_sha256
                )
            )
            for item in observations
            if item.provenance.repository_name is not None
        ),
        "pksim_uncertainty_semantics_explicit": all(
            item.provenance.uncertainty_auxiliary_type
            == "ArithmeticStdDev"
            for item in observations
            if item.provenance.repository_name is not None
            and item.source_uncertainty is not None
        ),
        "locked_observation_counts_exact": len(observations) == 50,
        "secondary_metric_count_exact": len(metrics) == 5,
    }
    payload = {
        "schema_version": "1.2.0",
        "builder_version": DATASET_BUILDER_VERSION,
        "dataset_version": DATASET_VERSION,
        "supersedes": "caffeine_dataset_v2",
        "change_reason": "REPRODUCIBLE_CANONICAL_MANIFEST_CHECKSUM",
        "generated_at": datetime.now(UTC),
        "source_artifacts": artifacts,
        "datasets": dataset_specs,
        "observation_count": len(observations),
        "metric_count": len(metrics),
        "split_counts": {
            split: sum(1 for item in observations if item.split == split)
            for split in (
                "calibration",
                "internal_holdout",
                "external_validation",
            )
        },
        "source_discrepancies": [
            {
                "code": (
                    "REPOSITORY_LABEL_DOSE_CONFLICT_RESOLVED_BY_SUPPLEMENT_"
                    "AND_PRIMARY_PUBLICATION"
                ),
                "detail": (
                    "The PK-Sim repository label says 0.5mgkg; Supinfo S2 "
                    "Table S3 and PMID 6832208 both state 5 mg/kg."
                ),
            },
            {
                "code": "SUPPLEMENT_POINT_COUNT_MISMATCH_10_VS_7",
                "detail": (
                    "Supinfo S2 says the Wahllander study has 10 time points; "
                    "the public PK-Sim observed repository contains 7. Only "
                    "the 7 stored values are retained."
                ),
            },
        ],
        "checks": checks,
        "accepted": all(checks.values()),
    }
    draft_manifest = CaffeineDatasetManifest(
        **payload,
        checksum="0" * 64,
    )
    manifest = draft_manifest.model_copy(
        update={
            "checksum": canonical_sha256(
                draft_manifest.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )
    if not manifest.accepted:
        failed = [name for name, passed in checks.items() if not passed]
        raise CaffeineDatasetError(
            "Caffeine dataset acceptance failed: " + ", ".join(failed)
        )

    output_hashes = {
        "observations.jsonl": _write_jsonl_atomic(
            output_dir / "observations.jsonl",
            observations,
        ),
        "pk_metrics.jsonl": _write_jsonl_atomic(
            output_dir / "pk_metrics.jsonl",
            metrics,
        ),
        "dataset_split_manifest.json": write_json_atomic(
            output_dir / "dataset_split_manifest.json",
            manifest,
            canonical=False,
        ),
    }
    write_json_atomic(
        output_dir / "files.sha256.json",
        output_hashes,
        canonical=True,
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the immutable, study-level caffeine calibration dataset.",
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pmc830-oa-archive", type=Path, required=True)
    parser.add_argument("--pmc830-nxml", type=Path, required=True)
    parser.add_argument("--pmc830-s001-zip", type=Path, required=True)
    parser.add_argument("--pmc830-pksim-project", type=Path, required=True)
    parser.add_argument("--pmc830-s002-docx", type=Path, required=True)
    parser.add_argument("--pmc535-nxml", type=Path, required=True)
    parser.add_argument("--pmc572-nxml", type=Path, required=True)
    parser.add_argument("--dimensions-xml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = build_caffeine_dataset(
        source_root=args.source_root,
        pmc830_oa_archive=args.pmc830_oa_archive,
        pmc830_nxml=args.pmc830_nxml,
        pmc830_s001_zip=args.pmc830_s001_zip,
        pmc830_pksim_project=args.pmc830_pksim_project,
        pmc830_s002_docx=args.pmc830_s002_docx,
        pmc535_nxml=args.pmc535_nxml,
        pmc572_nxml=args.pmc572_nxml,
        dimensions_xml=args.dimensions_xml,
        output_dir=args.output_dir,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
