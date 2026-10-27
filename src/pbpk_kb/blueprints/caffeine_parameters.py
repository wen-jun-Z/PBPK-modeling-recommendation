from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import fitz
from defusedxml import ElementTree as SafeET
from pydantic import Field, model_validator

from pbpk_kb.blueprints.schemas import StrictBlueprintModel
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file


PARAMETER_BUILDER_VERSION = "caffeine-parameter-builder-1.0.0"
PARAMETER_MANIFEST_VERSION = "caffeine_parameter_manifest_v1"
DATASET_VERSION = "caffeine_dataset_v3"
CAFFEINE_INCHIKEY = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
CALIBRATION_DATASETS = [
    "blanchard_1983_iv_5mgkg",
    "blanchard_1983_oral_5mgkg",
]
CALIBRATION_OBJECTIVE = (
    "0.5*mean((log(Cpred_iv)-log(Cobs_iv))^2)+"
    "0.5*mean((log(Cpred_oral)-log(Cobs_oral))^2)"
)


class CaffeineParameterError(RuntimeError):
    pass


class ParameterSourceArtifact(StrictBlueprintModel):
    artifact_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(gt=0)
    license: str = Field(min_length=1)
    redistribution: Literal[
        "public_reference",
        "noncommercial_no_derivatives",
        "metadata_only_no_redistribution",
        "osp_compatible_source",
    ]
    http_headers_location: str | None = None
    http_headers_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class ParameterProvenance(StrictBlueprintModel):
    artifact_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    extraction_method: Literal[
        "structured_json",
        "structured_xml",
        "pdf_text",
        "pdf_coordinate_table",
        "model_asset",
    ]
    source_value_text: str = Field(min_length=1)
    source_value_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DerivationInput(StrictBlueprintModel):
    subject: int = Field(ge=1)
    weight_kg: float = Field(gt=0)
    renal_clearance_ml_per_min: float = Field(gt=0)
    normalized_clearance_ml_per_min_per_kg: float = Field(gt=0)


class CaffeineParameterRecord(StrictBlueprintModel):
    parameter_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    pksim_path: str = Field(min_length=1)
    origin: Literal["observed", "derived", "estimated"]
    value: float
    unit: str = Field(min_length=1)
    uncertainty_type: Literal["sd"] | None = None
    uncertainty: float | None = Field(default=None, ge=0)
    reported_lower: float | None = None
    reported_upper: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    calibrate: bool
    calibration_dataset_ids: list[str] = Field(default_factory=list)
    objective_function: str | None = None
    derivation_expression: str | None = None
    derivation_inputs: list[DerivationInput] = Field(default_factory=list)
    provenance: list[ParameterProvenance] = Field(min_length=1)

    @model_validator(mode="after")
    def origin_contract(self) -> "CaffeineParameterRecord":
        if not math.isfinite(self.value):
            raise ValueError("Parameter value must be finite.")
        if self.reported_lower is not None and self.reported_upper is not None:
            if self.reported_lower > self.reported_upper:
                raise ValueError("Reported parameter range is reversed.")
        if self.origin == "estimated":
            if not self.calibrate:
                raise ValueError("Estimated parameters must be calibrated.")
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("Estimated parameters require locked bounds.")
            if not self.lower_bound <= self.value <= self.upper_bound:
                raise ValueError("Estimated parameter initial value is out of bounds.")
            if self.calibration_dataset_ids != CALIBRATION_DATASETS:
                raise ValueError("Estimated parameter calibration datasets drifted.")
            if self.objective_function != CALIBRATION_OBJECTIVE:
                raise ValueError("Estimated parameter objective drifted.")
        else:
            if self.calibrate or self.calibration_dataset_ids:
                raise ValueError("Observed/derived parameters cannot be calibrated.")
            if self.lower_bound is not None or self.upper_bound is not None:
                raise ValueError("Only estimated parameters define fit bounds.")
            if self.objective_function is not None:
                raise ValueError("Only estimated parameters define an objective.")
        if self.origin == "derived" and (
            not self.derivation_expression or not self.derivation_inputs
        ):
            raise ValueError("Derived parameters require expression and inputs.")
        if self.origin != "derived" and (
            self.derivation_expression or self.derivation_inputs
        ):
            raise ValueError("Only derived parameters carry derivation inputs.")
        if (self.uncertainty_type is None) != (self.uncertainty is None):
            raise ValueError("Uncertainty type and value must be provided together.")
        return self


class CaffeineParameterManifest(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    builder_version: Literal[PARAMETER_BUILDER_VERSION] = (
        PARAMETER_BUILDER_VERSION
    )
    manifest_version: Literal[PARAMETER_MANIFEST_VERSION] = (
        PARAMETER_MANIFEST_VERSION
    )
    generated_at: datetime
    dataset_version: Literal[DATASET_VERSION] = DATASET_VERSION
    dataset_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_artifacts: list[ParameterSourceArtifact] = Field(min_length=10)
    parameters: list[CaffeineParameterRecord] = Field(min_length=8, max_length=8)
    calibration_objective: Literal[CALIBRATION_OBJECTIVE] = CALIBRATION_OBJECTIVE
    structural_assumptions: list[str] = Field(min_length=3)
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


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
    source_url: str,
    license_name: str,
    redistribution: Literal[
        "public_reference",
        "noncommercial_no_derivatives",
        "metadata_only_no_redistribution",
        "osp_compatible_source",
    ],
    headers_path: Path | None = None,
) -> ParameterSourceArtifact:
    if not path.is_file() or path.stat().st_size <= 0:
        raise CaffeineParameterError(f"Source artifact is missing: {path}")
    if headers_path is not None and (
        not headers_path.is_file() or headers_path.stat().st_size <= 0
    ):
        raise CaffeineParameterError(
            f"HTTP header sidecar is missing: {headers_path}"
        )
    return ParameterSourceArtifact(
        artifact_id=artifact_id,
        role=role,
        source_url=source_url,
        source_location=_relative(path, root),
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        license=license_name,
        redistribution=redistribution,
        http_headers_location=(
            _relative(headers_path, root) if headers_path is not None else None
        ),
        http_headers_sha256=(
            sha256_file(headers_path) if headers_path is not None else None
        ),
    )


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(
    *,
    artifact_id: str,
    source_location: str,
    extraction_method: Literal[
        "structured_json",
        "structured_xml",
        "pdf_text",
        "pdf_coordinate_table",
        "model_asset",
    ],
    source_value_text: str,
) -> ParameterProvenance:
    return ParameterProvenance(
        artifact_id=artifact_id,
        source_location=source_location,
        extraction_method=extraction_method,
        source_value_text=source_value_text,
        source_value_text_sha256=_text_hash(source_value_text),
    )


def _walk_information(section: dict[str, Any]) -> list[dict[str, Any]]:
    values = list(section.get("Information", []))
    for child in section.get("Section", []):
        values.extend(_walk_information(child))
    return values


def _pubchem_information(path: Path, exact_text: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches: list[dict[str, Any]] = []
    for section in payload.get("Record", {}).get("Section", []):
        for information in _walk_information(section):
            strings = information.get("Value", {}).get("StringWithMarkup", [])
            if any(item.get("String") == exact_text for item in strings):
                matches.append(information)
    if len(matches) != 1:
        raise CaffeineParameterError(
            f"Expected one PubChem value {exact_text!r}; found {len(matches)}."
        )
    return matches[0]


def _extract_pubmed_kinetics(path: Path) -> dict[str, float | str]:
    root = SafeET.parse(path).getroot()
    articles = [
        item
        for item in root.findall(".//PubmedArticle")
        if item.findtext("./MedlineCitation/PMID") == "8553685"
    ]
    if len(articles) != 1:
        raise CaffeineParameterError("PubMed XML must contain PMID 8553685 once.")
    abstract_nodes = articles[0].findall(
        "./MedlineCitation/Article/Abstract/AbstractText"
    )
    abstract = " ".join(
        " ".join("".join(item.itertext()).split()) for item in abstract_nodes
    )
    pattern = re.compile(
        r"Km and Vmax.*?"
        r"(?P<km>\d+)\s*\(range\s*(?P<km_low>\d+)-(?P<km_high>\d+)\)"
        r"\s*microM,\s*and\s*(?P<vmax>\d+)\s*"
        r"\(range\s*(?P<vmax_low>\d+)-(?P<vmax_high>\d+)\)"
        r"\s*pmol\.mg protein-1\.min-1",
        re.IGNORECASE,
    )
    match = pattern.search(abstract)
    if match is None:
        raise CaffeineParameterError(
            "PMID 8553685 kinetic values do not match the locked statement."
        )
    values: dict[str, float | str] = {
        key: float(value) for key, value in match.groupdict().items()
    }
    values["source_text"] = match.group(0)
    return values


def _extract_lelo_fu(path: Path) -> tuple[float, float, str]:
    document = fitz.open(path)
    try:
        if document.page_count != 6:
            raise CaffeineParameterError("Lelo 1986 PDF page count drifted.")
        page_text = " ".join(document[2].get_text("text").split())
    finally:
        document.close()
    pattern = re.compile(
        r"mean free fractions of CA, PX, TP and TB were\s*"
        r"(?P<mean>0\.68)\s*(?:\+|\u00b1)\s*(?P<sd>0\.04)",
        re.IGNORECASE,
    )
    match = pattern.search(page_text)
    if match is None:
        raise CaffeineParameterError("Lelo 1986 caffeine fu statement is missing.")
    return float(match.group("mean")), float(match.group("sd")), match.group(0)


def _birkett_rows_from_words(
    words: list[tuple[float, float, float, float, str, int, int, int]],
) -> list[DerivationInput]:
    anchors = [
        item
        for item in words
        if 135 <= item[0] <= 150
        and 590 <= item[1] <= 730
        and item[4] in {"1", "2", "3", "4", "5", "6"}
    ]
    if len(anchors) != 6:
        raise CaffeineParameterError(
            f"Birkett Table 1 subject anchors drifted: {len(anchors)}."
        )
    rows: list[DerivationInput] = []
    for anchor in sorted(anchors, key=lambda item: int(item[4])):
        subject = int(anchor[4])
        y = anchor[1]
        weight_words = [
            item[4]
            for item in words
            if 190 <= item[0] <= 210
            and abs(item[1] - y) <= 4
            and re.fullmatch(r"\d+(?:\.\d+)?", item[4])
        ]
        clearance_words = [
            item[4]
            for item in words
            if 380 <= item[0] < 405
            and abs(item[1] - y) <= 4
            and re.fullmatch(r"\d+(?:\.\d+)?", item[4])
        ]
        if len(weight_words) != 1 or len(clearance_words) != 1:
            raise CaffeineParameterError(
                "Birkett Table 1 coordinate extraction is ambiguous for "
                f"subject {subject}: weight={weight_words}, CLR={clearance_words}."
            )
        weight = float(weight_words[0])
        clearance = float(clearance_words[0])
        rows.append(
            DerivationInput(
                subject=subject,
                weight_kg=weight,
                renal_clearance_ml_per_min=clearance,
                normalized_clearance_ml_per_min_per_kg=clearance / weight,
            )
        )
    if [item.subject for item in rows] != [1, 2, 3, 4, 5, 6]:
        raise CaffeineParameterError("Birkett Table 1 subjects are incomplete.")
    return rows


def _extract_birkett_renal_clearance(
    path: Path,
) -> tuple[float, list[DerivationInput], str]:
    document = fitz.open(path)
    try:
        if document.page_count != 4:
            raise CaffeineParameterError("Birkett 1991 PDF page count drifted.")
        page = document[1]
        page_text = page.get_text("text")
        if "Table 1" not in page_text or "caffeine pharmacokinetic parameters" not in page_text:
            raise CaffeineParameterError("Birkett 1991 Table 1 heading is missing.")
        rows = _birkett_rows_from_words(page.get_text("words"))
    finally:
        document.close()
    value = sum(
        item.normalized_clearance_ml_per_min_per_kg for item in rows
    ) / len(rows)
    source_text = json.dumps(
        [item.model_dump(mode="json") for item in rows],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return value, rows, source_text


def _verify_ncbi_identity_chain(
    idconv_path: Path,
    pubmed_path: Path,
    lelo_oa_path: Path,
    birkett_oa_path: Path,
) -> None:
    idconv = json.loads(idconv_path.read_text(encoding="utf-8"))
    mappings = {
        item["pmcid"]: (int(item["pmid"]), item["doi"])
        for item in idconv.get("records", [])
    }
    expected = {
        "PMC1401099": (3756065, "10.1111/j.1365-2125.1986.tb05246.x"),
        "PMC1368325": (2049248, "10.1111/j.1365-2125.1991.tb05553.x"),
    }
    if mappings != expected:
        raise CaffeineParameterError("NCBI PMID/PMCID/DOI mapping drifted.")
    root = SafeET.parse(pubmed_path).getroot()
    pmids = {
        int(value.text)
        for value in root.findall(".//MedlineCitation/PMID")
        if value.text
    }
    if pmids != {3756065, 2049248}:
        raise CaffeineParameterError("PubMed source identity set drifted.")
    for path, pmcid in (
        (lelo_oa_path, "PMC1401099"),
        (birkett_oa_path, "PMC1368325"),
    ):
        error = SafeET.parse(path).getroot().find("error")
        if error is None or error.attrib.get("code") != "idIsNotOpenAccess":
            raise CaffeineParameterError(
                f"PMC redistribution status drifted for {pmcid}."
            )


def _verify_dataset_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checksum = payload.pop("checksum", None)
    if payload.get("dataset_version") != DATASET_VERSION:
        raise CaffeineParameterError("Calibration dataset version drifted.")
    if payload.get("accepted") is not True:
        raise CaffeineParameterError("Calibration dataset is not accepted.")
    if canonical_sha256(payload) != checksum:
        raise CaffeineParameterError("Calibration dataset checksum is invalid.")
    datasets = {item["dataset_id"]: item for item in payload["datasets"]}
    if any(
        datasets.get(dataset_id, {}).get("split") != "calibration"
        for dataset_id in CALIBRATION_DATASETS
    ):
        raise CaffeineParameterError("Calibration dataset assignment drifted.")
    return {**payload, "checksum": checksum}


def _write_csv(path: Path, parameters: list[CaffeineParameterRecord]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "parameter_id",
            "origin",
            "value",
            "unit",
            "uncertainty",
            "reported_lower",
            "reported_upper",
            "lower_bound",
            "upper_bound",
            "calibrate",
            "pksim_path",
        ],
    )
    writer.writeheader()
    for item in parameters:
        writer.writerow(
            {
                key: getattr(item, key)
                for key in writer.fieldnames
            }
        )
    payload = buffer.getvalue().encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def build_caffeine_parameter_manifest(
    *,
    source_root: Path,
    dataset_manifest_path: Path,
    source_snapshot_path: Path,
    pubmed_kinetics_path: Path,
    pubchem_properties_path: Path,
    pubchem_pka_path: Path,
    pubchem_solubility_path: Path,
    lelo_pdf_path: Path,
    birkett_pdf_path: Path,
    ncbi_idconv_path: Path,
    pubmed_identity_path: Path,
    lelo_oa_response_path: Path,
    birkett_oa_response_path: Path,
    output_dir: Path,
) -> CaffeineParameterManifest:
    if output_dir.exists():
        raise CaffeineParameterError(
            f"Immutable parameter output exists: {output_dir}"
        )
    source_root = source_root.resolve()
    dataset = _verify_dataset_manifest(dataset_manifest_path)
    _verify_ncbi_identity_chain(
        ncbi_idconv_path,
        pubmed_identity_path,
        lelo_oa_response_path,
        birkett_oa_response_path,
    )

    source_snapshot = json.loads(source_snapshot_path.read_text(encoding="utf-8"))
    compounds = [
        item for item in source_snapshot.get("Compounds", [])
        if item.get("Name") == "Caffeine"
    ]
    if len(compounds) != 1:
        raise CaffeineParameterError("Source snapshot lacks one caffeine compound.")

    property_payload = json.loads(
        pubchem_properties_path.read_text(encoding="utf-8")
    )
    properties = property_payload.get("PropertyTable", {}).get("Properties", [])
    if len(properties) != 1 or int(properties[0].get("CID", -1)) != 2519:
        raise CaffeineParameterError("PubChem CID 2519 property record drifted.")
    prop = properties[0]
    if prop.get("InChIKey") != CAFFEINE_INCHIKEY:
        raise CaffeineParameterError("PubChem caffeine InChIKey drifted.")
    molecular_weight = float(prop["MolecularWeight"])
    xlogp = float(prop["XLogP"])

    pka_text = "pKa = 0.7 (caffeine cation)"
    pka_info = _pubchem_information(pubchem_pka_path, pka_text)
    if not any("Svorc L" in item for item in pka_info.get("Reference", [])):
        raise CaffeineParameterError("PubChem pKa reference drifted.")
    solubility_text = "In water, 2.16X10+4 mg/L at 25 \u00b0C"
    solubility_info = _pubchem_information(
        pubchem_solubility_path,
        solubility_text,
    )
    if not any("Yalkowsky" in item for item in solubility_info.get("Reference", [])):
        raise CaffeineParameterError("PubChem solubility reference drifted.")
    solubility = 2.16 * (10**4)

    kinetics = _extract_pubmed_kinetics(pubmed_kinetics_path)
    fu, fu_sd, fu_text = _extract_lelo_fu(lelo_pdf_path)
    renal, renal_rows, renal_text = _extract_birkett_renal_clearance(
        birkett_pdf_path
    )

    headers = lambda path: path.with_suffix(".headers")
    artifacts = [
        _artifact(
            dataset_manifest_path,
            root=source_root,
            artifact_id="caffeine_dataset_v3_manifest",
            role="locked_calibration_validation_split",
            source_url="internal:pbpk-model-assets/caffeine_dataset_v3",
            license_name="Derived from cited noncommercial research sources",
            redistribution="noncommercial_no_derivatives",
        ),
        _artifact(
            source_snapshot_path,
            root=source_root,
            artifact_id="caffeine_snapshot_v2",
            role="licensed_pksim_structure_snapshot",
            source_url="https://github.com/Open-Systems-Pharmacology/PK-Sim",
            license_name="GPL-2.0-or-later WITH OSP-Suite-Addendum",
            redistribution="osp_compatible_source",
        ),
        _artifact(
            pubmed_kinetics_path,
            root=source_root,
            artifact_id="pubmed_8553685_xml",
            role="cyp1a2_kinetics_primary_abstract",
            source_url=(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                "?db=pubmed&id=8553685&retmode=xml"
            ),
            license_name="NLM PubMed terms; metadata/abstract citation",
            redistribution="public_reference",
            headers_path=headers(pubmed_kinetics_path),
        ),
        _artifact(
            pubchem_properties_path,
            root=source_root,
            artifact_id="pubchem_2519_properties_json",
            role="compound_identity_and_computed_properties",
            source_url=(
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2519/"
                "property/MolecularFormula,MolecularWeight,XLogP,InChI,InChIKey/JSON"
            ),
            license_name="PubChem public data",
            redistribution="public_reference",
            headers_path=headers(pubchem_properties_path),
        ),
        _artifact(
            pubchem_pka_path,
            root=source_root,
            artifact_id="pubchem_2519_pka_json",
            role="pka_reference_record",
            source_url=(
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/"
                "2519/JSON?heading=Dissociation%20Constants"
            ),
            license_name="PubChem public data with cited source",
            redistribution="public_reference",
            headers_path=headers(pubchem_pka_path),
        ),
        _artifact(
            pubchem_solubility_path,
            root=source_root,
            artifact_id="pubchem_2519_solubility_json",
            role="aqueous_solubility_reference_record",
            source_url=(
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/"
                "2519/JSON?heading=Solubility"
            ),
            license_name="PubChem public data with cited source",
            redistribution="public_reference",
            headers_path=headers(pubchem_solubility_path),
        ),
        _artifact(
            lelo_pdf_path,
            root=source_root,
            artifact_id="lelo_1986_pdf",
            role="fraction_unbound_primary_article",
            source_url="https://pk-db.com/media/data/Lelo1986a.pdf",
            license_name="PMC-not-OA; internal noncommercial verification only",
            redistribution="metadata_only_no_redistribution",
            headers_path=headers(lelo_pdf_path),
        ),
        _artifact(
            birkett_pdf_path,
            root=source_root,
            artifact_id="birkett_1991_pdf",
            role="renal_clearance_primary_article",
            source_url="https://pk-db.com/media/data/Birkett1991.pdf",
            license_name="PMC-not-OA; internal noncommercial verification only",
            redistribution="metadata_only_no_redistribution",
            headers_path=headers(birkett_pdf_path),
        ),
        _artifact(
            ncbi_idconv_path,
            root=source_root,
            artifact_id="ncbi_idconv_lelo_birkett_json",
            role="pmid_pmcid_doi_identity_map",
            source_url=(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
                "?ids=PMC1368325,PMC1401099&format=json"
            ),
            license_name="NLM public metadata",
            redistribution="public_reference",
            headers_path=headers(ncbi_idconv_path),
        ),
        _artifact(
            pubmed_identity_path,
            root=source_root,
            artifact_id="pubmed_lelo_birkett_xml",
            role="primary_article_bibliographic_identity",
            source_url=(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                "?db=pubmed&id=2049248,3756065&retmode=xml"
            ),
            license_name="NLM PubMed terms; bibliographic metadata",
            redistribution="public_reference",
            headers_path=headers(pubmed_identity_path),
        ),
        _artifact(
            lelo_oa_response_path,
            root=source_root,
            artifact_id="pmc1401099_oa_status_xml",
            role="redistribution_status_evidence",
            source_url=(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
                "?id=PMC1401099"
            ),
            license_name="NLM OA status response",
            redistribution="public_reference",
            headers_path=headers(lelo_oa_response_path),
        ),
        _artifact(
            birkett_oa_response_path,
            root=source_root,
            artifact_id="pmc1368325_oa_status_xml",
            role="redistribution_status_evidence",
            source_url=(
                "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
                "?id=PMC1368325"
            ),
            license_name="NLM OA status response",
            redistribution="public_reference",
            headers_path=headers(birkett_oa_response_path),
        ),
    ]
    artifact_checksums = {item.artifact_id: item.sha256 for item in artifacts}

    def ref(
        artifact_id: str,
        source_location: str,
        method: Literal[
            "structured_json",
            "structured_xml",
            "pdf_text",
            "pdf_coordinate_table",
            "model_asset",
        ],
        source_value_text: str,
    ) -> list[ParameterProvenance]:
        if artifact_id not in artifact_checksums:
            raise CaffeineParameterError(f"Unknown provenance artifact: {artifact_id}")
        return [
            _provenance(
                artifact_id=artifact_id,
                source_location=source_location,
                extraction_method=method,
                source_value_text=source_value_text,
            )
        ]

    estimated = {
        "calibrate": True,
        "calibration_dataset_ids": CALIBRATION_DATASETS,
        "objective_function": CALIBRATION_OBJECTIVE,
    }
    parameters = [
        CaffeineParameterRecord(
            parameter_id="caffeine_molecular_weight",
            name="Molecular weight",
            pksim_path="Compounds|Caffeine|Parameters|Molecular weight",
            origin="observed",
            value=molecular_weight,
            unit="g/mol",
            calibrate=False,
            provenance=ref(
                "pubchem_2519_properties_json",
                "PropertyTable.Properties[CID=2519].MolecularWeight",
                "structured_json",
                str(prop["MolecularWeight"]),
            ),
        ),
        CaffeineParameterRecord(
            parameter_id="caffeine_pka_base",
            name="Base pKa",
            pksim_path="Compounds|Caffeine|PkaTypes|Base|Pka",
            origin="observed",
            value=0.7,
            unit="dimensionless",
            calibrate=False,
            provenance=ref(
                "pubchem_2519_pka_json",
                "Chemical and Physical Properties/Experimental Properties/"
                "Dissociation Constants[pKa;caffeine cation]",
                "structured_json",
                pka_text,
            ),
        ),
        CaffeineParameterRecord(
            parameter_id="caffeine_water_solubility_25c",
            name="Water solubility at 25 C",
            pksim_path=(
                "Compounds|Caffeine|Solubility|Water solubility|"
                "Solubility at reference pH"
            ),
            origin="observed",
            value=solubility,
            unit="mg/l",
            calibrate=False,
            provenance=ref(
                "pubchem_2519_solubility_json",
                "Chemical and Physical Properties/Experimental Properties/"
                "Solubility[Yalkowsky 2010 p.506]",
                "structured_json",
                solubility_text,
            ),
        ),
        CaffeineParameterRecord(
            parameter_id="caffeine_fraction_unbound_plasma",
            name="Fraction unbound in plasma",
            pksim_path=(
                "Compounds|Caffeine|FractionUnbound|Human|"
                "Fraction unbound (plasma, reference value)"
            ),
            origin="observed",
            value=fu,
            unit="fraction",
            uncertainty_type="sd",
            uncertainty=fu_sd,
            calibrate=False,
            provenance=ref(
                "lelo_1986_pdf",
                "PDF page 3, Results, caffeine mean free fraction",
                "pdf_text",
                fu_text,
            ),
        ),
        CaffeineParameterRecord(
            parameter_id="caffeine_renal_plasma_clearance",
            name="Renal plasma clearance",
            pksim_path=(
                "Compounds|Caffeine|Processes|KidneyClearance|Plasma clearance"
            ),
            origin="derived",
            value=renal,
            unit="ml/min/kg",
            calibrate=False,
            derivation_expression="mean(CLR_i_ml_per_min / weight_i_kg), i=1..6",
            derivation_inputs=renal_rows,
            provenance=ref(
                "birkett_1991_pdf",
                "PDF page 2, Table 1, Subject/Weight/CLR columns, rows 1-6",
                "pdf_coordinate_table",
                renal_text,
            ),
        ),
        CaffeineParameterRecord(
            parameter_id="caffeine_cyp1a2_km_n3_demethylation",
            name="CYP1A2 N3-demethylation Km",
            pksim_path="Compounds|Caffeine|Processes|CYP1A2-MM|Km",
            origin="observed",
            value=float(kinetics["km"]),
            unit="umol/l",
            reported_lower=float(kinetics["km_low"]),
            reported_upper=float(kinetics["km_high"]),
            calibrate=False,
            provenance=ref(
                "pubmed_8553685_xml",
                "PubmedArticle[PMID=8553685]/Abstract/AbstractText",
                "structured_xml",
                str(kinetics["source_text"]),
            ),
        ),
        CaffeineParameterRecord(
            parameter_id="caffeine_cyp1a2_vmax_effective_n3",
            name="CYP1A2 effective N3-demethylation Vmax",
            pksim_path=(
                "Compounds|Caffeine|Processes|CYP1A2-MM|"
                "In vitro Vmax for liver microsomes"
            ),
            origin="estimated",
            value=float(kinetics["vmax"]),
            unit="pmol/min/mg mic. protein",
            reported_lower=float(kinetics["vmax_low"]),
            reported_upper=float(kinetics["vmax_high"]),
            lower_bound=float(kinetics["vmax_low"]),
            upper_bound=float(kinetics["vmax_high"]),
            provenance=ref(
                "pubmed_8553685_xml",
                "PubmedArticle[PMID=8553685]/Abstract/AbstractText",
                "structured_xml",
                str(kinetics["source_text"]),
            ),
            **estimated,
        ),
        CaffeineParameterRecord(
            parameter_id="caffeine_effective_lipophilicity",
            name="Effective lipophilicity",
            pksim_path="Compounds|Caffeine|Lipophilicity|Lipophilicity",
            origin="estimated",
            value=xlogp,
            unit="Log Units",
            reported_lower=xlogp,
            reported_upper=xlogp,
            lower_bound=xlogp - 1.0,
            upper_bound=xlogp + 1.0,
            provenance=ref(
                "pubchem_2519_properties_json",
                "PropertyTable.Properties[CID=2519].XLogP",
                "structured_json",
                str(prop["XLogP"]),
            ),
            **estimated,
        ),
    ]

    estimated_ids = {
        item.parameter_id for item in parameters if item.origin == "estimated"
    }
    fixed_ids = {
        item.parameter_id for item in parameters if item.origin != "estimated"
    }
    validation_ids = {
        item["dataset_id"]
        for item in dataset["datasets"]
        if item["split"] != "calibration"
    }
    checks = {
        "dataset_manifest_checksum_reproducible": canonical_sha256(
            {
                key: value
                for key, value in dataset.items()
                if key != "checksum"
            }
        )
        == dataset["checksum"],
        "source_artifacts_nonempty_checksummed": all(
            item.byte_size > 0 and len(item.sha256) == 64 for item in artifacts
        ),
        "compound_identity_exact": prop["InChIKey"] == CAFFEINE_INCHIKEY,
        "parameter_ids_unique": len({item.parameter_id for item in parameters})
        == len(parameters),
        "only_vmax_and_lipophilicity_estimated": estimated_ids
        == {
            "caffeine_cyp1a2_vmax_effective_n3",
            "caffeine_effective_lipophilicity",
        },
        "fixed_parameters_not_calibrated": all(
            not item.calibrate
            for item in parameters
            if item.parameter_id in fixed_ids
        ),
        "calibration_excludes_holdout_and_external": not (
            set(CALIBRATION_DATASETS) & validation_ids
        ),
        "renal_clearance_six_subject_derivation": len(renal_rows) == 6
        and math.isclose(renal, 0.010330512664107847, rel_tol=0, abs_tol=1e-15),
        "kinetic_ranges_exact": kinetics["km_low"] == 220.0
        and kinetics["km_high"] == 1200.0
        and kinetics["vmax_low"] == 115.0
        and kinetics["vmax_high"] == 450.0,
        "restricted_pdfs_not_marked_redistributable": all(
            item.redistribution == "metadata_only_no_redistribution"
            for item in artifacts
            if item.artifact_id in {"lelo_1986_pdf", "birkett_1991_pdf"}
        ),
    }
    payload = {
        "schema_version": "1.0.0",
        "builder_version": PARAMETER_BUILDER_VERSION,
        "manifest_version": PARAMETER_MANIFEST_VERSION,
        "generated_at": datetime.now(UTC),
        "dataset_version": DATASET_VERSION,
        "dataset_manifest_checksum": dataset["checksum"],
        "source_artifacts": artifacts,
        "parameters": parameters,
        "calibration_objective": CALIBRATION_OBJECTIVE,
        "structural_assumptions": [
            (
                "The selected CYP1A2-MM process represents parent caffeine "
                "loss through the dominant N3-demethylation route; minor "
                "N1/N7 pathways and metabolites are not explicit in v1."
            ),
            (
                "PK-Sim Standard cellular permeability and partition methods "
                "are retained; no fitted intestinal permeability is copied."
            ),
            (
                "Only the two locked Blanchard calibration curves may update "
                "Vmax and effective lipophilicity; holdout/external data are "
                "evaluation-only."
            ),
        ],
        "checks": checks,
        "accepted": all(checks.values()),
    }
    draft = CaffeineParameterManifest(**payload, checksum="0" * 64)
    manifest = draft.model_copy(
        update={
            "checksum": canonical_sha256(
                draft.model_dump(mode="json", exclude={"checksum"})
            )
        }
    )
    if not manifest.accepted:
        failed = [name for name, passed in checks.items() if not passed]
        raise CaffeineParameterError(
            "Caffeine parameter acceptance failed: " + ", ".join(failed)
        )

    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise CaffeineParameterError(
            f"Stale parameter output exists: {temporary}"
        )
    temporary.mkdir(parents=True)
    write_json_atomic(
        temporary / "caffeine_parameter_manifest.json",
        manifest,
        canonical=False,
    )
    _write_csv(temporary / "parameter_values.csv", parameters)
    files = [
        {
            "file": path.name,
            "sha256": sha256_file(path),
            "byte_size": path.stat().st_size,
        }
        for path in sorted(temporary.iterdir(), key=lambda item: item.name)
        if path.is_file()
    ]
    write_json_atomic(
        temporary / "files.sha256.json",
        {"files": files},
        canonical=True,
    )
    temporary.replace(output_dir)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the provenance-locked caffeine parameter manifest."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--pubmed-kinetics", type=Path, required=True)
    parser.add_argument("--pubchem-properties", type=Path, required=True)
    parser.add_argument("--pubchem-pka", type=Path, required=True)
    parser.add_argument("--pubchem-solubility", type=Path, required=True)
    parser.add_argument("--lelo-pdf", type=Path, required=True)
    parser.add_argument("--birkett-pdf", type=Path, required=True)
    parser.add_argument("--ncbi-idconv", type=Path, required=True)
    parser.add_argument("--pubmed-identity", type=Path, required=True)
    parser.add_argument("--lelo-oa-response", type=Path, required=True)
    parser.add_argument("--birkett-oa-response", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = build_caffeine_parameter_manifest(
        source_root=args.source_root,
        dataset_manifest_path=args.dataset_manifest,
        source_snapshot_path=args.source_snapshot,
        pubmed_kinetics_path=args.pubmed_kinetics,
        pubchem_properties_path=args.pubchem_properties,
        pubchem_pka_path=args.pubchem_pka,
        pubchem_solubility_path=args.pubchem_solubility,
        lelo_pdf_path=args.lelo_pdf,
        birkett_pdf_path=args.birkett_pdf,
        ncbi_idconv_path=args.ncbi_idconv,
        pubmed_identity_path=args.pubmed_identity,
        lelo_oa_response_path=args.lelo_oa_response,
        birkett_oa_response_path=args.birkett_oa_response,
        output_dir=args.output_dir,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
