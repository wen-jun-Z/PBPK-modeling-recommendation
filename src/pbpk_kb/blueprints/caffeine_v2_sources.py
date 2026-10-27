from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from defusedxml import ElementTree as SafeET
from pydantic import Field

from pbpk_kb.blueprints.schemas import StrictBlueprintModel
from pbpk_kb.model_assets.execution_plan import (
    canonical_sha256,
    write_json_atomic,
)
from pbpk_kb.model_assets.license_policy import sha256_file


SOURCE_BUILDER_VERSION = "caffeine-v2-source-builder-1.0.0"
SOURCE_PACKAGE_VERSION = "caffeine_parameter_sources_v2"
LELO_PUBMED_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=pubmed&id=3756066&retmode=xml"
)
LELO_PDF_URL = "https://europepmc.org/api/getPdf?pmcid=PMC1401107"


class CaffeineV2SourceError(RuntimeError):
    pass


class SourceArtifact(StrictBlueprintModel):
    artifact_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(gt=0)
    license: str = Field(min_length=1)
    redistribution: Literal[
        "public_reference",
        "noncommercial_no_derivatives",
        "internal_reference_only",
        "derived_manifest",
    ]
    response_metadata_path: str | None = None
    response_metadata_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class PathwayFraction(StrictBlueprintModel):
    pathway_id: Literal["n3_paraxanthine", "n1_theobromine", "n7_theophylline"]
    fraction: float = Field(gt=0, lt=1)
    reported_sd: float = Field(gt=0)
    source_artifact_id: Literal["lelo_1986_partial_clearance_pubmed_xml"]
    source_location: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SearchDomain(StrictBlueprintModel):
    parameter_id: Literal[
        "total_cyp1a2_specific_clearance",
        "specific_intestinal_permeability_transcellular",
    ]
    unit: str = Field(min_length=1)
    lower_bound: Literal[0.0] = 0.0
    upper_bound: float = Field(gt=0)
    use: Literal["search_domain_only"] = "search_domain_only"
    source_artifact_id: Literal["fendt_2021_parameter_identification_xml"]
    source_location: str = Field(min_length=1)


class ContaminationRecord(StrictBlueprintModel):
    item: str = Field(min_length=1)
    decision: Literal["excluded_from_v2_model_input"]
    reason: str = Field(min_length=1)
    source_artifact_id: str = Field(min_length=1)


class CaffeineV2SourceManifest(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    builder_version: Literal[SOURCE_BUILDER_VERSION] = SOURCE_BUILDER_VERSION
    package_version: Literal[SOURCE_PACKAGE_VERSION] = SOURCE_PACKAGE_VERSION
    generated_at: datetime
    artifacts: list[SourceArtifact] = Field(min_length=6)
    pathway_fractions: list[PathwayFraction] = Field(min_length=3, max_length=3)
    search_domains: list[SearchDomain] = Field(min_length=2, max_length=2)
    contamination_ledger: list[ContaminationRecord] = Field(min_length=4)
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _response_metadata(response: object, requested_url: str) -> dict[str, object]:
    headers = getattr(response, "headers")
    return {
        "requested_url": requested_url,
        "final_url": getattr(response, "geturl")(),
        "status": getattr(response, "status"),
        "headers": {key.lower(): value for key, value in headers.items()},
        "downloaded_at": datetime.now(UTC).isoformat(),
    }


def _download(url: str, path: Path) -> Path:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PBPK-Stage1-KB/0.1 research-provenance acquisition"
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
        metadata = _response_metadata(response, url)
    if not payload:
        raise CaffeineV2SourceError(f"Downloaded an empty source: {url}")
    path.write_bytes(payload)
    metadata_path = path.with_suffix(path.suffix + ".response.json")
    write_json_atomic(metadata_path, metadata, canonical=True)
    return metadata_path


def _copy_required(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size <= 0:
        raise CaffeineV2SourceError(f"Required source is missing: {source}")
    shutil.copyfile(source, destination)


def _artifact(
    *,
    root: Path,
    path: Path,
    artifact_id: str,
    role: str,
    source_url: str,
    license_name: str,
    redistribution: Literal[
        "public_reference",
        "noncommercial_no_derivatives",
        "internal_reference_only",
        "derived_manifest",
    ],
    metadata_path: Path | None = None,
) -> SourceArtifact:
    return SourceArtifact(
        artifact_id=artifact_id,
        role=role,
        source_url=source_url,
        relative_path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        license=license_name,
        redistribution=redistribution,
        response_metadata_path=(
            metadata_path.relative_to(root).as_posix()
            if metadata_path is not None
            else None
        ),
        response_metadata_sha256=(
            sha256_file(metadata_path) if metadata_path is not None else None
        ),
    )


def _extract_pathway_fractions(path: Path) -> list[PathwayFraction]:
    root = SafeET.parse(path).getroot()
    articles = [
        item
        for item in root.findall(".//PubmedArticle")
        if item.findtext("./MedlineCitation/PMID") == "3756066"
    ]
    if len(articles) != 1:
        raise CaffeineV2SourceError("PubMed XML must contain PMID 3756066 once.")
    abstract = " ".join(
        " ".join("".join(node.itertext()).split())
        for node in articles[0].findall(
            "./MedlineCitation/Article/Abstract/AbstractText"
        )
    )
    match = re.search(
        r"PX, TB and TP accounted for\s*"
        r"(?P<n3>83\.9)\s*\+/-\s*(?P<n3_sd>5\.4)%,\s*"
        r"(?P<n1>12\.1)\s*\+/-\s*(?P<n1_sd>4\.1)%\s*and\s*"
        r"(?P<n7>4\.0)\s*\+/-\s*(?P<n7_sd>1\.4)%",
        abstract,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise CaffeineV2SourceError("Lelo pathway fraction statement drifted.")
    source_text = match.group(0)
    location = "PubmedArticle[PMID=3756066]/Abstract/AbstractText"
    specs = (
        ("n3_paraxanthine", "n3", "n3_sd"),
        ("n1_theobromine", "n1", "n1_sd"),
        ("n7_theophylline", "n7", "n7_sd"),
    )
    return [
        PathwayFraction(
            pathway_id=pathway_id,
            fraction=float(match.group(value_key)) / 100.0,
            reported_sd=float(match.group(sd_key)) / 100.0,
            source_artifact_id="lelo_1986_partial_clearance_pubmed_xml",
            source_location=location,
            source_text=source_text,
            source_text_sha256=_text_sha256(source_text),
        )
        for pathway_id, value_key, sd_key in specs
    ]


def _extract_search_domains(path: Path) -> list[SearchDomain]:
    root = SafeET.parse(path).getroot()
    parameters = {
        node.attrib.get("name"): node
        for node in root.findall(".//{*}IdentificationParameter")
    }
    expected = {
        "Specific clearance": (
            "total_cyp1a2_specific_clearance",
            "1/min",
            0.35848049857049,
        ),
        "Intestinal permeability (transcellular)": (
            "specific_intestinal_permeability_transcellular",
            "dm/min",
            3.85111864778853e-05,
        ),
    }
    output: list[SearchDomain] = []
    for source_name, (parameter_id, unit, expected_upper) in expected.items():
        node = parameters.get(source_name)
        if node is None:
            raise CaffeineV2SourceError(
                f"Fendt parameter-identification field is missing: {source_name}"
            )
        lower = float(node.attrib["minValue"])
        upper = float(node.attrib["maxValue"])
        if lower != 0.0 or not abs(upper - expected_upper) <= expected_upper * 1e-12:
            raise CaffeineV2SourceError(
                f"Fendt search domain drifted for {source_name}: {lower}, {upper}"
            )
        output.append(
            SearchDomain(
                parameter_id=parameter_id,
                unit=unit,
                upper_bound=upper,
                source_artifact_id="fendt_2021_parameter_identification_xml",
                source_location=(
                    "ParameterIdentification/IdentificationParameter"
                    f"[@name='{source_name}']/@minValue,@maxValue"
                ),
            )
        )
    return output


def build_caffeine_v2_source_package(
    *,
    v1_parameter_manifest_path: Path,
    fendt_article_nxml_path: Path,
    fendt_parameter_identification_path: Path,
    fendt_source_snapshot_path: Path,
    output_dir: Path,
) -> CaffeineV2SourceManifest:
    if output_dir.exists():
        raise CaffeineV2SourceError(f"Immutable output exists: {output_dir}")
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise CaffeineV2SourceError(f"Stale temporary output exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        lelo_xml = temporary / "lelo_1986_partial_clearance_pubmed.xml"
        lelo_pdf = temporary / "lelo_1986_partial_clearance.pdf"
        lelo_xml_metadata = _download(LELO_PUBMED_URL, lelo_xml)
        lelo_pdf_metadata = _download(LELO_PDF_URL, lelo_pdf)

        v1_manifest = temporary / "caffeine_parameter_manifest_v1.json"
        fendt_article = temporary / "fendt_2021_article.nxml"
        fendt_pi = temporary / "fendt_2021_parameter_identification.xml"
        fendt_snapshot = temporary / "fendt_2021_source_snapshot.json"
        _copy_required(v1_parameter_manifest_path, v1_manifest)
        _copy_required(fendt_article_nxml_path, fendt_article)
        _copy_required(fendt_parameter_identification_path, fendt_pi)
        _copy_required(fendt_source_snapshot_path, fendt_snapshot)

        v1_payload = json.loads(v1_manifest.read_text(encoding="utf-8"))
        v1_checksum = v1_payload.get("checksum")
        if canonical_sha256(
            {key: value for key, value in v1_payload.items() if key != "checksum"}
        ) != v1_checksum:
            raise CaffeineV2SourceError("v1 parameter manifest checksum is invalid.")

        article_root = SafeET.parse(fendt_article).getroot()
        license_refs = [
            "".join(node.itertext()).strip()
            for node in article_root.findall(".//{*}license_ref")
        ]
        if not any("by-nc-nd/4.0" in value for value in license_refs):
            raise CaffeineV2SourceError("Fendt CC BY-NC-ND license drifted.")

        fractions = _extract_pathway_fractions(lelo_xml)
        domains = _extract_search_domains(fendt_pi)
        artifacts = [
            _artifact(
                root=temporary,
                path=lelo_xml,
                artifact_id="lelo_1986_partial_clearance_pubmed_xml",
                role="human_demethylation_pathway_fractions",
                source_url=LELO_PUBMED_URL,
                license_name="NLM PubMed terms; abstract citation",
                redistribution="public_reference",
                metadata_path=lelo_xml_metadata,
            ),
            _artifact(
                root=temporary,
                path=lelo_pdf,
                artifact_id="lelo_1986_partial_clearance_pdf",
                role="primary_article_visual_verification",
                source_url=LELO_PDF_URL,
                license_name="PMC legacy full text; internal noncommercial verification",
                redistribution="internal_reference_only",
                metadata_path=lelo_pdf_metadata,
            ),
            _artifact(
                root=temporary,
                path=v1_manifest,
                artifact_id="caffeine_parameter_manifest_v1",
                role="audited_fixed_parameter_parent_manifest",
                source_url="internal:caffeine_parameter_manifest_v1",
                license_name="Derived provenance manifest",
                redistribution="derived_manifest",
            ),
            _artifact(
                root=temporary,
                path=fendt_article,
                artifact_id="fendt_2021_article_nxml",
                role="method_and_license_record",
                source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8302243/",
                license_name="CC BY-NC-ND 4.0",
                redistribution="noncommercial_no_derivatives",
            ),
            _artifact(
                root=temporary,
                path=fendt_pi,
                artifact_id="fendt_2021_parameter_identification_xml",
                role="search_domain_and_parameter_dimension_record",
                source_url=(
                    "PMC8302243 supplementary material PSP4-10-782-s001.zip/"
                    "Caffeine_BaseModel.xml"
                ),
                license_name="CC BY-NC-ND 4.0",
                redistribution="noncommercial_no_derivatives",
            ),
            _artifact(
                root=temporary,
                path=fendt_snapshot,
                artifact_id="fendt_2021_source_snapshot",
                role="internal_schema_and_contamination_audit",
                source_url=(
                    "PMC8302243 supplementary material PSP4-10-782-s001.zip/"
                    "Caffeine_50mg_PO.pksim5"
                ),
                license_name="CC BY-NC-ND 4.0",
                redistribution="internal_reference_only",
            ),
        ]
        contamination = [
            ContaminationRecord(
                item="Fendt model fitted CYP1A2 specific clearance value",
                decision="excluded_from_v2_model_input",
                reason="Parameter-identification result is development-data dependent.",
                source_artifact_id="fendt_2021_source_snapshot",
            ),
            ContaminationRecord(
                item="Fendt model fitted intestinal permeability value",
                decision="excluded_from_v2_model_input",
                reason="Fitted alternative is development-data dependent.",
                source_artifact_id="fendt_2021_source_snapshot",
            ),
            ContaminationRecord(
                item="Fendt model prediction curves and validation conclusions",
                decision="excluded_from_v2_model_input",
                reason="Predictions cannot become parameter or validation evidence.",
                source_artifact_id="fendt_2021_article_nxml",
            ),
            ContaminationRecord(
                item="Fendt CC BY-NC-ND model file and snapshot topology",
                decision="excluded_from_v2_model_input",
                reason=(
                    "v2 topology is generated from the GPL PK-Sim schema; the ND model "
                    "is retained only for internal schema and leakage inspection."
                ),
                source_artifact_id="fendt_2021_source_snapshot",
            ),
        ]
        checks = {
            "v1_manifest_checksum_valid": isinstance(v1_checksum, str),
            "pathway_fraction_sum_one": abs(
                sum(item.fraction for item in fractions) - 1.0
            )
            <= 1e-12,
            "pathway_fraction_order_locked": [
                item.pathway_id for item in fractions
            ]
            == ["n3_paraxanthine", "n1_theobromine", "n7_theophylline"],
            "search_domains_exact": len(domains) == 2,
            "fendt_license_noncommercial_no_derivatives": any(
                "by-nc-nd/4.0" in value for value in license_refs
            ),
            "fendt_fitted_values_excluded": len(contamination) == 4
            and all(
                item.decision == "excluded_from_v2_model_input"
                for item in contamination
            ),
            "artifacts_nonempty_and_checksummed": all(
                item.byte_size > 0 and len(item.sha256) == 64
                for item in artifacts
            ),
        }
        payload = {
            "schema_version": "1.0.0",
            "builder_version": SOURCE_BUILDER_VERSION,
            "package_version": SOURCE_PACKAGE_VERSION,
            "generated_at": datetime.now(UTC),
            "artifacts": artifacts,
            "pathway_fractions": fractions,
            "search_domains": domains,
            "contamination_ledger": contamination,
            "checks": checks,
            "accepted": all(checks.values()),
        }
        draft = CaffeineV2SourceManifest(**payload, checksum="0" * 64)
        manifest = draft.model_copy(
            update={
                "checksum": canonical_sha256(
                    draft.model_dump(mode="json", exclude={"checksum"})
                )
            }
        )
        if not manifest.accepted:
            failed = [name for name, passed in checks.items() if not passed]
            raise CaffeineV2SourceError(
                "Caffeine v2 source package failed: " + ", ".join(failed)
            )
        write_json_atomic(
            temporary / "source_manifest.json", manifest, canonical=False
        )
        write_json_atomic(
            temporary / "contamination_ledger.json",
            [item.model_dump(mode="json") for item in contamination],
            canonical=False,
        )
        files = [
            {
                "relative_path": path.relative_to(temporary).as_posix(),
                "sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
            }
            for path in sorted(temporary.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.name != "files.sha256.json"
        ]
        write_json_atomic(
            temporary / "files.sha256.json", {"files": files}, canonical=True
        )
        temporary.replace(output_dir)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the immutable caffeine v2 scientific source package."
    )
    parser.add_argument("--v1-parameter-manifest", type=Path, required=True)
    parser.add_argument("--fendt-article-nxml", type=Path, required=True)
    parser.add_argument("--fendt-parameter-identification", type=Path, required=True)
    parser.add_argument("--fendt-source-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = build_caffeine_v2_source_package(
        v1_parameter_manifest_path=args.v1_parameter_manifest,
        fendt_article_nxml_path=args.fendt_article_nxml,
        fendt_parameter_identification_path=args.fendt_parameter_identification,
        fendt_source_snapshot_path=args.fendt_source_snapshot,
        output_dir=args.output_dir,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
