"""Source-bound scientific context; no network, generated facts or database writes."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re
from typing import Any

from defusedxml import ElementTree as SafeET

from pbpk_kb.parsers.table_structure import StructuredTable, element_text, local_name

CONTEXT_VERSION = "source-context-v2-body-row-boundary"
SPECIES = {
    "rat": r"\b(?:rats?|rattus|rodents?)\b",
    "mouse": r"\b(?:mouse|mice|murine|mus musculus)\b",
    "dog": r"\b(?:dogs?|canine|beagles?)\b",
    "cat": r"\b(?:cats?|feline)\b",
    "monkey": r"\b(?:monkeys?|macaques?|macaca|cynomolgus|primates?)\b",
    "rabbit": r"\b(?:rabbits?|oryctolagus)\b",
    "pig": r"\b(?:pigs?|porcine|swine|minipigs?)\b",
}
PARAMETER_HEADER = re.compile(
    r"\b(?:C\s*max|AUC(?:\s*(?:t|inf|∞|\d+(?:\s*[-–]\s*(?:t|inf|∞|\d+))?\s*h?))?|T\s*max|half[- ]life|clearance|CL(?:/F)?|"
    r"Vd|Vz|Vss|volume of distribution|bioavailability|protein binding|fraction unbound|"
    r"bound\s+to\s+(?:(?:human|plasma|serum)\s+)*proteins?|excreted unchanged)\b|t\s*1/2|t½", re.I)
METABOLITE_HEADER = re.compile(r"\b(?:active\s+)?metabolites?\b", re.I)


def repair_multilevel_headers(table: StructuredTable) -> StructuredTable:
    """Join only leading header tiers before an identifiable PK parameter row.

    The existing XML parser expands rowspan/colspan. Some SPLs encode later
    header tiers with td, so those tiers previously appeared as body rows.
    Numeric/population rows of a table whose columns already name endpoints
    are never promoted by this function.
    """
    if any(PARAMETER_HEADER.search(c) for c in table.columns):
        return table
    first_data = None
    for index, row in enumerate(table.rows[:10]):
        if any(PARAMETER_HEADER.search(cell) for cell in row) and any(
                re.match(r"^\s*(?:[<>≤≥~≈]=?\s*|(?:greater than(?: or equal to)?|"
                         r"less than(?: or equal to)?|at least|at most|more than|"
                         r"approximately|around|about)\s+)?\d", cell, re.I) for cell in row):
            first_data = index
            break
    if first_data is None or first_data == 0:
        return table
    tiers = table.rows[:first_data]
    # A descriptive PK body row can precede numeric rows (for example a food
    # effect mentioning Cmax and AUC). It must not contaminate every header.
    if any(PARAMETER_HEADER.search(cell) for row in tiers for cell in row):
        return table
    if any(len(row) != len(table.columns) for row in tiers):
        return table
    # A full-width subtitle (e.g. a wrapped table title) is not a population
    # assignment for every subsequent adult/child column.
    spanning_title = len(set(table.columns)) == 1 and len(table.columns) > 1
    columns = []
    for index, old in enumerate(table.columns):
        parts = [old] if old and not spanning_title and not old.lower().startswith("column_") else []
        parts += [tier[index] for tier in tiers if tier[index]]
        columns.append(" / ".join(dict.fromkeys(parts)) or f"column_{index + 1}")
    caption = " / ".join(dict.fromkeys(value for value in (table.caption, table.columns[0] if spanning_title else None) if value)) or None
    return StructuredTable(columns=columns, rows=table.rows[first_data:], caption=caption,
                           footnotes=table.footnotes, metadata={**table.metadata,
                           "restored_header_rows": tiers, "context_version": CONTEXT_VERSION})


def condition_fields(text: str) -> dict[str, str | None]:
    """Extract verbatim conditions from the selected column/row only."""
    value = re.sub(r"\s+", " ", text).strip()
    patterns = {
        "dose": r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:mg|µg|μg|ug|mcg|g)(?:\s*/\s*kg)?\b",
        "route": r"\b(?:intravenously|intravenous|orally|oral|subcutaneous|intramuscular|buccal|sublingual|transdermal)\b",
        "population": r"\b(?:healthy (?:adults|volunteers|subjects)|adults?|children|pediatric|paediatric|neonates?|elderly|geriatric|pregnant|age\s+\d+\s+to\s+\d+)\b",
        "food_state": r"\b(?:high[ -]fat[^/;]{0,45}?fed state|low[ -]fat[^/;]{0,45}?fed state|fasted|fasting|fed|with food|without food)\b",
        "formulation": r"\b(?:(?:extended|sustained|immediate|controlled)[ -]release(?:\s+(?:tablets?|capsules?))?|tablets?|capsules?|solution|suspension)\b",
        "dosing_interval": r"\b(?:every\s+\d+\s+hours?|once daily|twice daily|three times daily|q\d+h)\b",
    }
    result = {}
    for key, pattern in patterns.items():
        candidate = value
        if key == "dose":
            outside_parentheses = re.sub(r"\([^)]*\)", " ", value)
            if re.search(pattern, outside_parentheses, re.I):
                candidate = outside_parentheses
        matches = list(dict.fromkeys(m.group(0) for m in re.finditer(pattern, candidate, re.I)))
        result[key] = matches[0] if len(matches) == 1 else None
    return result


def analyte_mismatch_reason(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata_json") or {}
    if metadata.get("analyte_role") == "metabolite":
        return "metabolite_value_not_parent_drug"
    if row.get("evidence_type") not in {"clinical_pk", "adme"} or row.get("value") is None:
        return None
    if not re.search(r"clearance|\bCL\b", str(row.get("parameter") or ""), re.I):
        return None
    source = str(row.get("source_text") or "")
    target = str(row.get("attributed_compound_name") or row.get("compound_name") or row.get("compound") or "")
    if not target or not re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", source, re.I):
        return None
    # Require an explicit grammatical object, not capitalized words anywhere in
    # the passage. E.g. ibuprofen increased 'clearance of IS and pCS up to ...'.
    match = re.search(r"\bclearance\s+of\s+(.{1,85}?)\s+(?:up\s+to|was|were|is|increased|decreased|by|from|after|during|with|following|relative)\b", source, re.I)
    if not match:
        return None
    obj = match.group(1).strip(" ,;:")
    if re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", obj, re.I):
        return None
    if re.search(r"\b(?:the drug|this drug|drug|compound|parent|dose|plasma|body|creatinine|it)\b", obj, re.I):
        return None
    # Only a named analyte, including commonly reported solute abbreviations,
    # can establish a mismatch; numeric or generic descriptors remain unknown.
    if re.fullmatch(r"(?:IS|pCS|indoxyl sulfate|p-cresyl sulfate)(?:\s+and\s+(?:IS|pCS|indoxyl sulfate|p-cresyl sulfate))?", obj, re.I):
        return "clearance_value_belongs_to_other_named_analyte"
    return None


def _normalized(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def species_mentions(value: str) -> list[str]:
    # CAT/PIG can be all-capital clinical abbreviations. The extractor must not
    # infer species merely because a scientific sentence contains capitals.
    return [name for name, pattern in SPECIES.items()
            if any(not match.group(0).isupper() for match in re.finditer(pattern, value, re.I))]


def _article_context(path: Path, checksum: str, quote: str, cache: dict) -> dict[str, Any]:
    key = (str(path), checksum)
    if key not in cache:
        data = path.read_bytes()
        if not checksum or sha256(data).hexdigest() != checksum:
            raise ValueError("Source checksum does not match registered evidence")
        root = SafeET.fromstring(data)
        articles = []
        for node in root.iter():
            if local_name(node.tag) not in {"article", "PubmedArticle"}:
                continue
            title = next((element_text(n) for n in node.iter() if local_name(n.tag) in {"article-title", "ArticleTitle"}), "")
            abstracts = [element_text(n) for n in node.iter() if local_name(n.tag) in {"abstract", "Abstract"}]
            mesh = [element_text(n) for n in node.iter() if local_name(n.tag) == "MeshHeading"]
            articles.append((_normalized(element_text(node)), {"title": title, "abstract": " ".join(abstracts), "mesh": mesh}))
        # Bound memory for batched XML responses; cached evidence does not need
        # the entire 70-compound literature corpus resident at once.
        if len(cache) >= 12:
            cache.pop(next(iter(cache)))
        cache[key] = articles
    normalized = _normalized(quote)
    matches = [info for article, info in cache[key] if len(normalized) >= 30 and normalized in article]
    unique = {repr(info): info for info in matches}
    if len(unique) != 1:
        return {"match_status": "unresolved" if not unique else "ambiguous"}
    info = next(iter(unique.values()))
    content = " ".join([info["title"], info["abstract"], *info["mesh"]])
    species = species_mentions(content)
    human_mesh = any(re.search(r"\bhumans\b", term, re.I) for term in info["mesh"])
    human_context = bool(human_mesh or re.search(r"\b(?:humans?|patients?|volunteers?)\b", content, re.I))
    title_species = species_mentions(info["title"])
    return {**info, "match_status": "matched", "species": species,
            "human_mesh": human_mesh, "human_context": human_context,
            "species_scope": "mixed_or_comparative" if species and human_context else "animal" if species else "unspecified_or_human",
            "animal_study": bool(title_species or (species and not human_mesh)),
            "title_species": title_species}


def enrich_source_context(row: dict[str, Any], *, project_root: Path, cache: dict | None = None) -> dict[str, Any]:
    """Return a copy with source-supported context; leave the source quote/hash intact.

    Only PubMed/PMC XML is inspected here. The caller supplies a cache to avoid
    rereading/checksumming large assets once per evidence item. Ambiguous article
    matches never borrow another article's species or study conditions.
    """
    result = deepcopy(row)
    if row.get("source_id") not in {"pubmed", "pmc_oa"}:
        return result
    path = Path(str(row.get("storage_path") or ""))
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_relative_to(project_root.resolve()):
        raise ValueError("Evidence context source escapes the project")
    if path.suffix.lower() not in {".xml", ".nxml"}:
        return result
    metadata = dict(result.get("metadata_json") or {})
    info = _article_context(path, str(row.get("checksum") or ""), str(row.get("source_text") or ""), cache if cache is not None else {})
    metadata["source_context"] = {**info, "version": CONTEXT_VERSION, "source_checksum": row.get("checksum"),
                                  "source_location": row.get("source_location")}
    if info.get("animal_study") and not info.get("human_context") and not result.get("population"):
        result["population"] = "; ".join(info["species"])
        metadata["population_context_method"] = "matched_article_title_abstract_mesh"
    result["metadata_json"] = metadata
    return result
