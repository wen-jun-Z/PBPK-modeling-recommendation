#!/usr/bin/env python3
"""Index local, traceable PBPK assessment sources without producing gold labels.

Only frozen candidate identities/provenance and their colocated source documents
are used. Ranking scores, legacy labels and model acceptance flags are not read.
Document statements remain author-reported claims, not independently validated
prediction errors. Existing parsed model contents are linked separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments/model_recommendation/frontier_comparison_20260914"
SCHEMA = "pbpk-assessment-sources-v1"
MAX_DOC_BYTES = 8 * 1024 * 1024
MAX_SEGMENT_CHARS = 1800
MAX_SEGMENTS_PER_DOCUMENT = 100
MAX_SEGMENTS_PER_CANDIDATE = 240
SKIP_SECTION = re.compile(r"code of conduct|contribution|license|acknowledg", re.I)
INTEREST = re.compile(
    r"PBPK|pharmacokinetic|model|simulat|validat|verif|predict|observ|AUC|Cmax|"
    r"clinical|healthy|adult|pediatr|pregnan|renal|hepatic|lactat|oral|intravenous|"
    r"enzyme|transporter|CYP|UGT|P-gp|OATP|clearance|dissolu|absorp|distribution|"
    r"dose|dosing|formulation|study|studies|data|GMFE|RMSE|MPE|reference|DOI", re.I)
METHOD_ONLY = re.compile(
    r"acceptance criteri|criterion|considered acceptable|used to (?:assess|evaluat)|"
    r"will be|should be|dashed lines|dotted lines|lines (?:depict|represent)|"
    r"(?:plot|figure)s? (?:show|present|display)|(?:was|were) (?:calculated|computed)", re.I)
VALIDATION_CLAIM = re.compile(
    r"(?:model|simulat\w*|predict\w*|fit).{0,130}(?:adequate|accurate|good agreement|"
    r"within|captur|reproduc|satisfactor|successfully|describ|validated|verified)|"
    r"(?:adequate|accurate|good|satisfactor).{0,60}(?:description|agreement|predict)|"
    r"(?:validated|verified|evaluated) (?:against|by compar|using|with)", re.I | re.S)
QUANTITATIVE_CLAIM = re.compile(
    r"\b(?:predicted|simulated|predictions|simulations)\b.{0,110}\b(?:within|deviat\w*|error\w*)\b"
    r".{0,40}(?:\d+(?:\.\d+)?|two)[ -]*(?:fold|%|percent)|"
    r"\b(?:GMFE|RMSE|AFE|AAFE|MPE|mean (?:fold |prediction )?error)\b"
    r".{0,45}(?:was|were|=|:|<|>|≤|≥)\s*\d|"
    r"\d+(?:\.\d+)?\s*%.{0,70}\b(?:predictions|predicted|simulated)\b.{0,50}\bwithin\b|"
    r"\b(?:predicted.to.observed|simulated.to.observed)\b.{0,40}\bratios?\b.{0,40}"
    r"\b(?:was|were|ranged|between)\b.{0,25}\d", re.I | re.S)
LIMITATION = re.compile(
    r"\b(?:cannot|unable|not suitable|not applicable|not validated|not evaluated|"
    r"not recommended|limited to|limitation|overpredict|underpredict|poor agreement|"
    r"failed to|inaccurate|not for large molecules)\b", re.I)
HOLDOUT = re.compile(
    r"not (?:been )?used (?:for|in) (?:model )?(?:building|development|fitting|calibration)|"
    r"independent (?:clinical )?(?:validation|verification|data|studies)|"
    r"external (?:validation|verification)|held[- ]out", re.I)
TAGS = {
    "population": {
        "healthy": r"\bhealthy\b", "adult": r"\badults?\b", "pediatric": r"pediatr|paediatr|children|infants?|neonat",
        "pregnancy": r"pregnan|maternal|fetal|foetal|gestation", "lactation": r"lactat|breast.?milk|breastfeed",
        "renal_impairment": r"renal (?:impair|insufficien)|kidney disease|hemodialysis|haemodialysis",
        "hepatic_impairment": r"hepatic (?:impair|insufficien)|liver disease|cirrhosis",
    },
    "route": {"oral": r"\boral\b|\bPO\b", "intravenous": r"intravenous|\bIV\b|\bi\.v\.",
              "subcutaneous": r"subcutaneous", "intramuscular": r"intramuscular", "inhalation": r"inhal(?:ation|ed)"},
    "endpoint": {"auc": r"\bAUC\b|area under", "cmax": r"C[_ ]?max|peak (?:plasma |blood )?concentration",
                 "concentration_time": r"concentration.?time|plasma (?:concentration|profile)|time.?profile",
                 "ddi": r"\bDDI\b|drug.?drug interaction", "urinary_excretion": r"urin|renal excretion"},
    "mechanism": {"cyp3a4": r"CYP3A4", "cyp2d6": r"CYP2D6", "cyp2c9": r"CYP2C9", "cyp1a2": r"CYP1A2",
                  "p_gp": r"P-gp|P glycoprotein|ABCB1", "oatp": r"OATP", "ugt": r"UGT", "renal": r"renal|glomerular"},
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def local_path(root: Path, value: str | Path) -> Path:
    """Reject traversal and symlinks outside the authorized project."""
    path = (root / value).resolve()
    path.relative_to(root.resolve())
    return path


def classify(excerpt: str, section: str = "") -> dict:
    method = bool(METHOD_ONLY.search(excerpt))
    metrics = reported_table_metrics(excerpt)
    quantitative = bool(QUANTITATIVE_CLAIM.search(excerpt) or metrics) and not method
    reported = bool(VALIDATION_CLAIM.search(excerpt)) and not method
    limitation = bool(LIMITATION.search(excerpt))
    if re.search(r"references", section, re.I):
        kind = "bibliographic_reference"
        reported = quantitative = False
        metrics = []
    elif limitation:
        kind = "limitation"
    elif quantitative:
        kind = "reported_quantitative_performance"
    elif reported:
        kind = "reported_validation_statement"
    elif method or re.search(r"diagnostic|goodness.of.fit", section, re.I):
        kind = "evaluation_method_only"
    else:
        kind = "model_description"
    return {
        "evidence_kind": kind,
        "flags": {"author_reported_not_recomputed": bool(reported or quantitative),
                  "external_holdout_explicit_in_excerpt": bool(HOLDOUT.search(excerpt)),
                  "validation_keyword_alone_is_not_quality": True},
        "reported_metrics": metrics,
        "tags": {group: [name for name, pattern in values.items() if re.search(pattern, excerpt, re.I)]
                 for group, values in TAGS.items()},
    }


def reported_table_metrics(excerpt: str) -> list[dict]:
    """Extract numeric report tables only, retaining their study/group qualifier."""
    header = None
    output = []
    known = {"gmfe", "rmse", "afe", "aafe", "mpe"}
    for line in excerpt.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [part.strip().strip("*") for part in line.strip().strip("|").split("|")]
        if any(cell.casefold() in known for cell in cells):
            header = cells
            continue
        if header is None or len(cells) != len(header):
            continue
        for index, name in enumerate(header):
            if name.casefold() not in known or not re.fullmatch(r"\d+(?:\.\d+)?", cells[index]):
                continue
            group = " | ".join(value for i, value in enumerate(cells) if i != index)
            partition = ("model_building" if re.search(r"model building|training|calibration", group, re.I)
                         else "model_verification_reported" if re.search(r"model verification|validation", group, re.I)
                         else "unspecified")
            output.append({"metric": name.upper(), "value": float(cells[index]), "group": group,
                           "evaluation_partition": partition, "author_reported": True,
                           "comparable_across_different_study_groups": False})
    return output


def text_segments(text: str, *, path: str, file_sha: str, page: int | None = None,
                  initial_section: str = "") -> list[dict]:
    """Keep exact text spans, source locations and contextual section headings."""
    lines = text.splitlines(keepends=True)
    sections: dict[int, str] = {}
    section = initial_section
    offset = 0
    spans = []
    pending_start = None
    pending_end = 0
    pending_line = 0
    pending_section = ""

    def flush(end_line):
        nonlocal pending_start
        if pending_start is not None:
            spans.append((pending_start, pending_end, pending_line, end_line, pending_section))
        pending_start = None

    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        header = re.match(r"^#{1,6}\s+(.+)", stripped)
        if header:
            flush(number - 1)
            section = re.sub(r"<[^>]+>", "", header.group(1)).strip()
        sections[number] = section
        if not stripped:
            flush(number - 1)
        else:
            if pending_start is not None and pending_end - pending_start + len(line) > MAX_SEGMENT_CHARS:
                flush(number - 1)
            if pending_start is None:
                pending_start = offset
                pending_line = number
                pending_section = section
            pending_end = offset + len(line)
        offset += len(line)
    flush(len(lines))
    selected = []
    for start, end, first_line, last_line, section in spans:
        if SKIP_SECTION.search(section):
            continue
        raw = text[start:end]
        if not INTEREST.search(raw):
            continue
        if re.fullmatch(r"\s*(?:!\[.*?\]\(.*?\)|<img\b[^>]*>)\s*", raw, re.S):
            continue
        # Long single-line paragraphs are split without inventing or cleaning text.
        for piece_start in range(start, end, MAX_SEGMENT_CHARS):
            piece_end = min(piece_start + MAX_SEGMENT_CHARS, end)
            excerpt = text[piece_start:piece_end]
            if not excerpt.strip():
                continue
            location = {"line_start": first_line, "line_end": last_line,
                        "char_start": piece_start, "char_end": piece_end, "section": section}
            if page is not None:
                location["page"] = page
                location["character_offsets_are_page_text"] = True
            evidence_id = "source:" + sha256_bytes(f"{file_sha}:{page}:{piece_start}:{piece_end}".encode())[:24]
            selected.append({"evidence_id": evidence_id, "location": location, "excerpt": excerpt,
                             "excerpt_sha256": sha256_bytes(excerpt.encode()),
                             "paragraph_split": end - start > MAX_SEGMENT_CHARS,
                             **classify(excerpt, section)})
    return selected


def source_directory(source: Path, root: Path) -> tuple[Path, str]:
    relative = source.relative_to(root)
    if "source" in relative.parts:
        number = relative.parts.index("source")
        return root.joinpath(*relative.parts[:number + 1]), "same_repository_snapshot"
    return source.parent, "same_library_directory"


def document_paths(folder: Path, root: Path) -> list[Path]:
    found = []
    for path in folder.rglob("*"):
        if path.suffix.lower() not in {".md", ".pdf"}:
            continue
        local_path(root, path)
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue
        if path.name.lower() in {"license.md", "contributing.md", "code_of_conduct.md"}:
            continue
        found.append(path)
    return sorted(found, key=lambda p: (p.suffix.lower() == ".pdf", str(p)))


def load_document(path: Path, root: Path) -> dict:
    relative = str(path.relative_to(root))
    size = path.stat().st_size
    digest = sha256_file(path)
    entry = {"path": relative, "sha256": digest, "bytes": size, "segments": []}
    if path.suffix.lower() == ".pdf" and path.with_suffix(".md").exists():
        return {**entry, "status": "markdown_counterpart_used",
                "counterpart": str(path.with_suffix(".md").relative_to(root))}
    if size > MAX_DOC_BYTES:
        return {**entry, "status": "skipped_size_limit"}
    if path.suffix.lower() == ".pdf":
        # Most library PDFs have Markdown counterparts; avoid duplicating claims.
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            segments = []
            for page_number, page in enumerate(reader.pages, 1):
                segments.extend(text_segments(page.extract_text() or "", path=relative, file_sha=digest, page=page_number))
        except Exception as exc:
            return {**entry, "status": "pdf_extraction_error", "error_type": type(exc).__name__}
    else:
        text = path.read_text(encoding="utf-8")
        segments = text_segments(text, path=relative, file_sha=digest, initial_section=path.stem)
    priority = {"reported_quantitative_performance": 0, "limitation": 1,
                "reported_validation_statement": 2, "model_description": 3,
                "evaluation_method_only": 4, "bibliographic_reference": 5}
    ranked = sorted(enumerate(segments), key=lambda pair: (priority[pair[1]["evidence_kind"]], pair[0]))
    retained = sorted(ranked[:MAX_SEGMENTS_PER_DOCUMENT])
    return {**entry, "status": "extracted", "segments": [s for _, s in retained],
            "eligible_segments": len(segments), "omitted_segments": len(segments) - len(retained),
            "extraction": "pdf_text_no_ocr" if path.suffix.lower() == ".pdf" else "exact_utf8_spans"}


def build(root: Path = ROOT, out: Path = DEFAULT_OUT) -> dict:
    root = root.resolve()
    out = local_path(root, out)
    candidates_file = out / "inputs/candidates.jsonl"
    candidates = [json.loads(line) for line in candidates_file.read_text().splitlines() if line.strip()]
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate frozen candidate IDs")
    catalog_file = root / "experiments/model_recommendation/model_expansion_20260909/catalog_candidates.json"
    catalog = {}
    if catalog_file.exists():
        for index, row in enumerate(json.loads(catalog_file.read_text())):
            catalog[(row["project_relative_path"], row["project_sha256"])] = (index, row)
    catalog_sha = sha256_file(catalog_file) if catalog_file.exists() else None
    cache = {}
    folder_cache = {}
    records = []
    candidate_counts = Counter()
    for candidate in candidates:
        provenance = candidate["provenance"]
        source = local_path(root, provenance["source_path"])
        if not source.is_file():
            raise ValueError(f"Missing frozen source: {source.relative_to(root)}")
        actual_sha = sha256_file(source)
        expected_sha = provenance["source_sha256"]
        if actual_sha != expected_sha:
            raise ValueError(f"Frozen candidate source SHA mismatch: {source.relative_to(root)}")
        folder, scope = source_directory(source, root)
        if str(folder) not in folder_cache:
            folder_cache[str(folder)] = document_paths(folder, root)
        paths = folder_cache[str(folder)]
        documents = []
        remaining = MAX_SEGMENTS_PER_CANDIDATE
        for path in paths:
            key = str(path)
            if key not in cache:
                cache[key] = load_document(path, root)
            original = cache[key]
            segments = original["segments"][:remaining]
            remaining -= len(segments)
            documents.append({**original, "scope": scope, "segments": segments,
                              "candidate_segment_limit_omitted": len(original["segments"]) - len(segments),
                              "version_specific_validation_verified": False,
                              "shared_repository_document": len([c for c in candidates if c["provenance"]["source_path"].startswith(str(folder.relative_to(root)) + "/")]) > 1})
        kinds = Counter(s["evidence_kind"] for d in documents for s in d["segments"])
        parser_info = catalog.get((provenance["source_path"], actual_sha))
        structure = {"evidence_role": "descriptive_only_not_independent_gold",
                     "frozen_input_location": f"inputs/candidates.jsonl#candidate_id={candidate['candidate_id']}",
                     "features": candidate["features"]}
        if parser_info:
            index, parsed = parser_info
            structure.update({"existing_parser_source": str(catalog_file.relative_to(root)),
                              "existing_parser_source_sha256": catalog_sha,
                              "json_pointer": f"/{index}", "parser_version": parsed.get("parser_version"),
                              "source_pointers": parsed.get("source_pointers", {}),
                              "simulation_count": parsed.get("simulation_count"),
                              "dataset_reference_count": parsed.get("dataset_reference_count")})
        record = {"schema": SCHEMA, "candidate_id": candidate["candidate_id"],
                  "family_id": candidate["family_id"], "family_name": candidate["family_name"],
                  "compounds": candidate["compounds"],
                  "source_artifact": {"path": provenance["source_path"], "sha256": actual_sha,
                                      "expected_sha256": expected_sha, "verified": True,
                                      "repository_commit": provenance.get("repository_commit")},
                  "documents": documents, "parsed_structure": structure,
                  "coverage": {"document_count": len(documents), "segment_count": sum(kinds.values()),
                               "evidence_kinds": dict(kinds),
                               "document_extraction_complete": all(d.get("status") in {"extracted", "markdown_counterpart_used"} and not d.get("omitted_segments", 0) and not d.get("candidate_segment_limit_omitted", 0) for d in documents),
                               "scientific_applicability": "not_adjudicated",
                               "no_document_means": "unknown_not_negative"}}
        records.append(record)
        candidate_counts["with_documents"] += bool(documents)
        candidate_counts["with_reported_quantitative_claim"] += bool(kinds["reported_quantitative_performance"])
        candidate_counts["with_reported_validation_statement"] += bool(kinds["reported_validation_statement"])
        candidate_counts["with_limitation_statement"] += bool(kinds["limitation"])
        candidate_counts["with_explicit_holdout_text"] += any(s["flags"]["external_holdout_explicit_in_excerpt"] for d in documents for s in d["segments"])
    destination = out / "assessment"
    destination.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records)
    output_file = destination / "sources.jsonl"
    temporary = output_file.with_suffix(".jsonl.tmp")
    temporary.write_text(payload)
    temporary.replace(output_file)
    manifest = {"schema": SCHEMA, "generated_at": datetime.now(UTC).isoformat(),
                "candidate_count": len(records), "source_artifact_hashes_verified": len(records),
                "candidate_coverage": dict(candidate_counts), "unique_document_count": len(cache),
                "unique_document_status_counts": dict(Counter(d["status"] for d in cache.values())),
                "unique_document_evidence_kind_counts": dict(Counter(s["evidence_kind"] for d in cache.values() for s in d["segments"])),
                "inputs": {"frozen_candidates": {"path": str(candidates_file.relative_to(root)), "sha256": sha256_file(candidates_file)},
                           "existing_parser_catalog": {"path": str(catalog_file.relative_to(root)), "sha256": catalog_sha}},
                "output": {"path": str(output_file.relative_to(root)), "sha256": sha256_file(output_file), "bytes": output_file.stat().st_size},
                "code_sha256": sha256_file(Path(__file__)),
                "limits": {"max_document_bytes": MAX_DOC_BYTES, "max_excerpt_characters": MAX_SEGMENT_CHARS,
                           "max_segments_per_document": MAX_SEGMENTS_PER_DOCUMENT,
                           "max_segments_per_candidate": MAX_SEGMENTS_PER_CANDIDATE},
                "policy": {"ranking_scores_read": False, "legacy_labels_read": False,
                           "documents_selected_by_model_performance": False,
                           "labels_created": False, "human_adjudicated": False,
                           "author_claims_are_not_measured_prediction_errors": True,
                           "colocation_does_not_establish_version_specific_validation": True,
                           "images_not_interpreted_as_numerical_results": True,
                           "descriptive_features_not_independent_scientific_answers": True}}
    manifest_file = destination / "sources_manifest.json"
    temporary = manifest_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(manifest_file)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(build(out=args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
