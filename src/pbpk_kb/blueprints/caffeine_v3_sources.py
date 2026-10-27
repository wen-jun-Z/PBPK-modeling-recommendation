from __future__ import annotations

import argparse
import hashlib
import html
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


SOURCE_BUILDER_VERSION = "caffeine-v3-source-builder-1.0.0"
SOURCE_PACKAGE_VERSION = "caffeine_absorption_sources_v3"
PKSIM_COMPOUND_DOC_URL = (
    "https://docs.open-systems-pharmacology.org/working-with-pk-sim/"
    "pk-sim-documentation/pk-sim-compounds-definition-and-work-flow"
)
OSP_ISSUE_URL = (
    "https://api.github.com/repos/Open-Systems-Pharmacology/Forum/issues/23"
)
OSP_COMMENTS_URL = OSP_ISSUE_URL + "/comments?per_page=100"
PUBMED_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=pubmed&id={pmid}&retmode=xml"
)
PMC_FULL_TEXT_URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/"
    "PMC3734790/fullTextXML"
)


class CaffeineV3SourceError(RuntimeError):
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
        "open_access_cc_by",
        "derived_manifest",
    ]
    response_metadata_path: str | None = None
    response_metadata_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class AbsorptionEvidence(StrictBlueprintModel):
    evidence_id: str = Field(min_length=1)
    evidence_type: Literal[
        "method_constraint",
        "transport_mechanism",
        "caco2_permeability_range",
        "absolute_bioavailability",
        "tmax",
        "human_high_permeability",
    ]
    use: Literal[
        "prohibit_direct_mapping",
        "structure_only",
        "absorption_constraint",
    ]
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    unit: str | None = None
    source_artifact_id: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CaffeineV3SourceManifest(StrictBlueprintModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    builder_version: Literal[SOURCE_BUILDER_VERSION] = SOURCE_BUILDER_VERSION
    package_version: Literal[SOURCE_PACKAGE_VERSION] = SOURCE_PACKAGE_VERSION
    generated_at: datetime
    parent_parameter_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_parameter_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: list[SourceArtifact] = Field(min_length=7, max_length=7)
    evidence: list[AbsorptionEvidence] = Field(min_length=7)
    prohibited_transforms: list[str] = Field(min_length=3)
    checks: dict[str, bool]
    accepted: bool
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", " ", value))
    value = value.replace("\u2212", "-").replace("\u2013", "-")
    value = value.replace("\u00b1", "+/-").replace("\u00d7", "x")
    return " ".join(value.split())


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
            "Accept": "application/json, application/xml, text/html;q=0.9",
            "User-Agent": "PBPK-Stage1-KB/0.1 research-provenance acquisition",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
        metadata = _response_metadata(response, url)
    if not payload:
        raise CaffeineV3SourceError(f"Downloaded an empty source: {url}")
    path.write_bytes(payload)
    metadata_path = path.with_suffix(path.suffix + ".response.json")
    write_json_atomic(metadata_path, metadata, canonical=True)
    return metadata_path


def _copy_verified_manifest(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file() or source.stat().st_size <= 0:
        raise CaffeineV3SourceError(f"Parent manifest is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("manifest_version") != "caffeine_parameter_manifest_v2":
        raise CaffeineV3SourceError("Parent parameter version drifted.")
    checksum = payload.get("checksum")
    if not isinstance(checksum, str) or canonical_sha256(
        {key: value for key, value in payload.items() if key != "checksum"}
    ) != checksum:
        raise CaffeineV3SourceError("Parent parameter checksum is invalid.")
    shutil.copyfile(source, destination)
    return payload


def _copy_prefetched_source(
    *,
    source: Path,
    destination: Path,
    source_url: str,
) -> Path:
    if not source.is_file() or source.stat().st_size <= 0:
        raise CaffeineV3SourceError(f"Prefetched source is missing: {source}")
    shutil.copyfile(source, destination)
    metadata_path = destination.with_suffix(destination.suffix + ".response.json")
    write_json_atomic(
        metadata_path,
        {
            "requested_url": source_url,
            "final_url": source_url,
            "status": 200,
            "headers": {},
            "downloaded_at": datetime.now(UTC).isoformat(),
            "acquisition_method": "prefetched_public_response",
            "prefetched_source_sha256": sha256_file(source),
        },
        canonical=True,
    )
    return metadata_path


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
        "open_access_cc_by",
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


def _evidence(
    *,
    evidence_id: str,
    evidence_type: Literal[
        "method_constraint",
        "transport_mechanism",
        "caco2_permeability_range",
        "absolute_bioavailability",
        "tmax",
        "human_high_permeability",
    ],
    use: Literal[
        "prohibit_direct_mapping",
        "structure_only",
        "absorption_constraint",
    ],
    source_artifact_id: str,
    source_location: str,
    source_text: str,
    value: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    unit: str | None = None,
) -> AbsorptionEvidence:
    normalized = _normalize_text(source_text)
    return AbsorptionEvidence(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        use=use,
        value=value,
        lower=lower,
        upper=upper,
        unit=unit,
        source_artifact_id=source_artifact_id,
        source_location=source_location,
        source_text=normalized,
        source_text_sha256=_text_hash(normalized),
    )


def _pubmed_article(path: Path, pmid: str) -> tuple[object, str]:
    root = SafeET.parse(path).getroot()
    articles = [
        item
        for item in root.findall(".//PubmedArticle")
        if item.findtext("./MedlineCitation/PMID") == pmid
    ]
    if len(articles) != 1:
        raise CaffeineV3SourceError(
            f"PubMed XML must contain PMID {pmid} exactly once."
        )
    nodes = articles[0].findall(
        "./MedlineCitation/Article/Abstract/AbstractText"
    )
    abstract = " ".join("".join(node.itertext()) for node in nodes)
    return articles[0], _normalize_text(abstract)


def extract_caffeine_absorption_evidence(
    *,
    pksim_doc_path: Path,
    osp_comments_path: Path,
    caco2_pubmed_path: Path,
    bioavailability_pubmed_path: Path,
    human_permeability_path: Path,
) -> list[AbsorptionEvidence]:
    doc_text = _normalize_text(pksim_doc_path.read_text(encoding="utf-8"))
    if not (
        "calibration of these in vitro values has to be performed" in doc_text
        and "specific intestinal permeability" in doc_text.lower()
    ):
        raise CaffeineV3SourceError("PK-Sim permeability guidance drifted.")

    comments = json.loads(osp_comments_path.read_text(encoding="utf-8"))
    comment_text = _normalize_text(
        " ".join(
            str(item.get("body", ""))
            for item in comments
            if isinstance(item, dict)
        )
    )
    comment_text_folded = comment_text.casefold()
    if not (
        "pint from pk-sim only makes sense in pk-sim" in comment_text_folded
        and "problem with unit conversion" in comment_text_folded
    ):
        raise CaffeineV3SourceError("OSP model-specific guidance drifted.")

    _, caco2 = _pubmed_article(caco2_pubmed_path, "20027153")
    caco_match = re.search(
        r"AP-BL direction Papp\s*=\s*(46\.3)-(53\.5)\s*x\s*10-6\s*cm/s",
        caco2,
        flags=re.IGNORECASE,
    )
    reverse_match = re.search(
        r"BL-AP direction Papp\s*=\s*(45\.6)-(49\.4)\s*x\s*10-6\s*cm/s",
        caco2,
        flags=re.IGNORECASE,
    )
    if caco_match is None or reverse_match is None:
        raise CaffeineV3SourceError("Caffeine Caco-2 ranges drifted.")
    mechanism_match = re.search(
        r"transport seems to be transcellular mediated by passive diffusion",
        caco2,
        flags=re.IGNORECASE,
    )
    if mechanism_match is None:
        raise CaffeineV3SourceError("Caffeine transport mechanism drifted.")

    _, bioavailability = _pubmed_article(bioavailability_pubmed_path, "6832208")
    f_match = re.search(
        r"F\(%\)\s*=\s*(108\.3)\s*\+/-\s*(3\.6)%",
        bioavailability,
        flags=re.IGNORECASE,
    )
    tmax_match = re.search(
        r"peak \(Tp\) plasma concentration after\s*(29\.8)\s*\+/-\s*"
        r"(8\.1)\s*min",
        bioavailability,
        flags=re.IGNORECASE,
    )
    if f_match is None or tmax_match is None:
        raise CaffeineV3SourceError("Blanchard absorption endpoints drifted.")

    human_root = SafeET.parse(human_permeability_path).getroot()
    human_text = _normalize_text(" ".join(human_root.itertext()))
    human_match = re.search(
        r"caffeine, acetaminophen and antipyrine have very high "
        r"permeabilities \(about 20 x 10\s*-4\s*cm/sec\)",
        human_text,
        flags=re.IGNORECASE,
    )
    if human_match is None:
        raise CaffeineV3SourceError("Human high-permeability statement drifted.")

    return [
        _evidence(
            evidence_id="pksim_in_vitro_values_require_calibration",
            evidence_type="method_constraint",
            use="prohibit_direct_mapping",
            source_artifact_id="pksim_compound_workflow_html",
            source_location="Absorption/Specific Intestinal Permeability",
            source_text=(
                "Experimental in vitro permeability values require calibration "
                "for a defined compound set before PK-Sim input."
            ),
        ),
        _evidence(
            evidence_id="pksim_pint_is_model_specific",
            evidence_type="method_constraint",
            use="prohibit_direct_mapping",
            source_artifact_id="osp_forum_issue_23_comments_json",
            source_location="comment by Aedginto, 2017-06-22",
            source_text="Pint from PK-Sim only makes sense in PK-Sim.",
        ),
        _evidence(
            evidence_id="caffeine_caco2_ap_bl_high",
            evidence_type="caco2_permeability_range",
            use="structure_only",
            source_artifact_id="smetanova_2009_pubmed_xml",
            source_location=(
                "PubmedArticle[PMID=20027153]/Abstract/RESULTS"
            ),
            source_text=caco_match.group(0),
            lower=float(caco_match.group(1)),
            upper=float(caco_match.group(2)),
            unit="10^-6 cm/s",
        ),
        _evidence(
            evidence_id="caffeine_caco2_passive_transcellular",
            evidence_type="transport_mechanism",
            use="structure_only",
            source_artifact_id="smetanova_2009_pubmed_xml",
            source_location=(
                "PubmedArticle[PMID=20027153]/Abstract/RESULTS"
            ),
            source_text=mechanism_match.group(0),
        ),
        _evidence(
            evidence_id="blanchard_absolute_bioavailability",
            evidence_type="absolute_bioavailability",
            use="absorption_constraint",
            source_artifact_id="blanchard_1983_pubmed_xml",
            source_location="PubmedArticle[PMID=6832208]/Abstract",
            source_text=f_match.group(0),
            value=float(f_match.group(1)),
            unit="percent (reported mean)",
        ),
        _evidence(
            evidence_id="blanchard_oral_tmax",
            evidence_type="tmax",
            use="absorption_constraint",
            source_artifact_id="blanchard_1983_pubmed_xml",
            source_location="PubmedArticle[PMID=6832208]/Abstract",
            source_text=tmax_match.group(0),
            value=float(tmax_match.group(1)),
            unit="min (reported mean; SEM 8.1; n=10)",
        ),
        _evidence(
            evidence_id="human_caffeine_high_permeability",
            evidence_type="human_high_permeability",
            use="structure_only",
            source_artifact_id="levitt_2013_full_text_xml",
            source_location="article/abstract/sec[Results]/p",
            source_text=human_match.group(0),
            value=20.0,
            unit="10^-4 cm/s (AM permeability; structure support only)",
        ),
    ]


def build_caffeine_v3_source_package(
    *,
    v2_parameter_manifest_path: Path,
    output_dir: Path,
    prefetched_osp_issue_path: Path | None = None,
    prefetched_osp_comments_path: Path | None = None,
) -> CaffeineV3SourceManifest:
    if output_dir.exists():
        raise CaffeineV3SourceError(f"Immutable output exists: {output_dir}")
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise CaffeineV3SourceError(f"Stale temporary output exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        paths = {
            "pksim": temporary / "pksim_compound_workflow.html",
            "issue": temporary / "osp_forum_issue_23.json",
            "comments": temporary / "osp_forum_issue_23_comments.json",
            "caco2": temporary / "smetanova_2009_pubmed.xml",
            "bioavailability": temporary / "blanchard_1983_pubmed.xml",
            "human": temporary / "levitt_2013_full_text.xml",
        }
        issue_metadata = (
            _copy_prefetched_source(
                source=prefetched_osp_issue_path,
                destination=paths["issue"],
                source_url=OSP_ISSUE_URL,
            )
            if prefetched_osp_issue_path is not None
            else _download(OSP_ISSUE_URL, paths["issue"])
        )
        comments_metadata = (
            _copy_prefetched_source(
                source=prefetched_osp_comments_path,
                destination=paths["comments"],
                source_url=OSP_COMMENTS_URL,
            )
            if prefetched_osp_comments_path is not None
            else _download(OSP_COMMENTS_URL, paths["comments"])
        )
        metadata = {
            "pksim": _download(PKSIM_COMPOUND_DOC_URL, paths["pksim"]),
            "issue": issue_metadata,
            "comments": comments_metadata,
            "caco2": _download(PUBMED_URL.format(pmid="20027153"), paths["caco2"]),
            "bioavailability": _download(
                PUBMED_URL.format(pmid="6832208"), paths["bioavailability"]
            ),
            "human": _download(PMC_FULL_TEXT_URL, paths["human"]),
        }
        parent_path = temporary / "caffeine_parameter_manifest_v2.json"
        parent = _copy_verified_manifest(v2_parameter_manifest_path, parent_path)

        evidence = extract_caffeine_absorption_evidence(
            pksim_doc_path=paths["pksim"],
            osp_comments_path=paths["comments"],
            caco2_pubmed_path=paths["caco2"],
            bioavailability_pubmed_path=paths["bioavailability"],
            human_permeability_path=paths["human"],
        )
        artifacts = [
            _artifact(
                root=temporary,
                path=paths["pksim"],
                artifact_id="pksim_compound_workflow_html",
                role="official_model_parameter_definition",
                source_url=PKSIM_COMPOUND_DOC_URL,
                license_name="Open Systems Pharmacology public documentation",
                redistribution="public_reference",
                metadata_path=metadata["pksim"],
            ),
            _artifact(
                root=temporary,
                path=paths["issue"],
                artifact_id="osp_forum_issue_23_json",
                role="official_issue_identity_and_context",
                source_url=OSP_ISSUE_URL,
                license_name="GitHub public issue metadata",
                redistribution="public_reference",
                metadata_path=metadata["issue"],
            ),
            _artifact(
                root=temporary,
                path=paths["comments"],
                artifact_id="osp_forum_issue_23_comments_json",
                role="official_model_specific_permeability_guidance",
                source_url=OSP_COMMENTS_URL,
                license_name="GitHub public issue discussion",
                redistribution="public_reference",
                metadata_path=metadata["comments"],
            ),
            _artifact(
                root=temporary,
                path=paths["caco2"],
                artifact_id="smetanova_2009_pubmed_xml",
                role="caffeine_transport_mechanism_and_caco2_range",
                source_url=PUBMED_URL.format(pmid="20027153"),
                license_name="NLM PubMed terms; abstract citation",
                redistribution="public_reference",
                metadata_path=metadata["caco2"],
            ),
            _artifact(
                root=temporary,
                path=paths["bioavailability"],
                artifact_id="blanchard_1983_pubmed_xml",
                role="absolute_bioavailability_and_tmax_constraint",
                source_url=PUBMED_URL.format(pmid="6832208"),
                license_name="NLM PubMed terms; abstract citation",
                redistribution="public_reference",
                metadata_path=metadata["bioavailability"],
            ),
            _artifact(
                root=temporary,
                path=paths["human"],
                artifact_id="levitt_2013_full_text_xml",
                role="human_high_permeability_structure_support",
                source_url=PMC_FULL_TEXT_URL,
                license_name="CC BY 2.0",
                redistribution="open_access_cc_by",
                metadata_path=metadata["human"],
            ),
            _artifact(
                root=temporary,
                path=parent_path,
                artifact_id="caffeine_parameter_manifest_v2",
                role="audited_parent_parameter_manifest",
                source_url="internal:caffeine_parameter_manifest_v2",
                license_name="Derived provenance manifest",
                redistribution="derived_manifest",
            ),
        ]
        prohibited = [
            "Do not convert Caco-2 Papp one-to-one into PK-Sim specific permeability.",
            "Do not copy Fendt fitted permeability into the v3 model.",
            "Do not use locked validation curves to define the permeability set.",
        ]
        evidence_by_id = {item.evidence_id: item for item in evidence}
        checks = {
            "parent_v2_parameter_manifest_valid": parent.get("accepted") is True,
            "official_direct_mapping_prohibition_present": (
                evidence_by_id[
                    "pksim_in_vitro_values_require_calibration"
                ].use
                == "prohibit_direct_mapping"
            ),
            "caco2_range_structure_only": (
                evidence_by_id["caffeine_caco2_ap_bl_high"].lower == 46.3
                and evidence_by_id["caffeine_caco2_ap_bl_high"].upper == 53.5
                and evidence_by_id["caffeine_caco2_ap_bl_high"].use
                == "structure_only"
            ),
            "blanchard_constraints_exact": (
                evidence_by_id["blanchard_absolute_bioavailability"].value
                == 108.3
                and evidence_by_id["blanchard_oral_tmax"].value == 29.8
            ),
            "human_permeability_structure_only": (
                evidence_by_id["human_caffeine_high_permeability"].use
                == "structure_only"
            ),
            "artifacts_nonempty_and_checksummed": all(
                item.byte_size > 0 and len(item.sha256) == 64
                for item in artifacts
            ),
            "all_downloads_have_response_metadata": all(
                item.response_metadata_path is not None
                for item in artifacts
                if item.redistribution != "derived_manifest"
            ),
        }
        payload = {
            "schema_version": "1.0.0",
            "builder_version": SOURCE_BUILDER_VERSION,
            "package_version": SOURCE_PACKAGE_VERSION,
            "generated_at": datetime.now(UTC),
            "parent_parameter_manifest_sha256": sha256_file(parent_path),
            "parent_parameter_manifest_checksum": parent["checksum"],
            "artifacts": artifacts,
            "evidence": evidence,
            "prohibited_transforms": prohibited,
            "checks": checks,
            "accepted": all(checks.values()),
        }
        draft = CaffeineV3SourceManifest(**payload, checksum="0" * 64)
        manifest = draft.model_copy(
            update={
                "checksum": canonical_sha256(
                    draft.model_dump(mode="json", exclude={"checksum"})
                )
            }
        )
        if not manifest.accepted:
            failed = [name for name, passed in checks.items() if not passed]
            raise CaffeineV3SourceError(
                "Caffeine v3 source package failed: " + ", ".join(failed)
            )
        write_json_atomic(
            temporary / "source_manifest.json", manifest, canonical=False
        )
        write_json_atomic(
            temporary / "absorption_evidence.json",
            [item.model_dump(mode="json") for item in evidence],
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
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the immutable caffeine v3 absorption source package."
    )
    parser.add_argument("--v2-parameter-manifest", type=Path, required=True)
    parser.add_argument("--prefetched-osp-issue", type=Path)
    parser.add_argument("--prefetched-osp-comments", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = build_caffeine_v3_source_package(
        v2_parameter_manifest_path=args.v2_parameter_manifest,
        output_dir=args.output_dir,
        prefetched_osp_issue_path=args.prefetched_osp_issue,
        prefetched_osp_comments_path=args.prefetched_osp_comments,
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
