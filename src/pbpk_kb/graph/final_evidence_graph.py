from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TargetMention:
    key: str
    preferred_name: str
    gene_symbol: str | None
    target_type: str
    matched_text: str


@dataclass(frozen=True)
class GraphFact:
    target: TargetMention
    relationship_type: str
    evidence_sentence: str
    confidence: float


STATIC_TARGETS: tuple[tuple[str, str, str | None, str, tuple[str, ...]], ...] = (
    ("ABCB1", "P-glycoprotein", "ABCB1", "transporter", ("P-glycoprotein", "P glycoprotein", "P-gp", "MDR1", "ABCB1")),
    ("ABCG2", "Breast cancer resistance protein", "ABCG2", "transporter", ("BCRP", "ABCG2")),
    ("SLCO1B1", "OATP1B1", "SLCO1B1", "transporter", ("OATP1B1", "SLCO1B1")),
    ("SLCO1B3", "OATP1B3", "SLCO1B3", "transporter", ("OATP1B3", "SLCO1B3")),
    ("SLCO2B1", "OATP2B1", "SLCO2B1", "transporter", ("OATP2B1", "SLCO2B1")),
    ("SLCO1A2", "OATP1A2", "SLCO1A2", "transporter", ("OATP1A2", "SLCO1A2")),
    ("OATP1A4", "OATP1A4", None, "transporter", ("OATP1A4", "Oatp1a4")),
    ("SLC22A1", "Organic cation transporter 1", "SLC22A1", "transporter", ("OCT1", "hOCT1", "SLC22A1", "organic cation transporter 1")),
    ("SLC22A2", "Organic cation transporter 2", "SLC22A2", "transporter", ("OCT2", "hOCT2", "SLC22A2", "organic cation transporter 2")),
    ("SLC22A3", "Organic cation transporter 3", "SLC22A3", "transporter", ("OCT3", "hOCT3", "SLC22A3", "organic cation transporter 3")),
    ("SLC22A6", "Organic anion transporter 1", "SLC22A6", "transporter", ("OAT1", "SLC22A6", "organic anion transporter 1")),
    ("SLC22A8", "Organic anion transporter 3", "SLC22A8", "transporter", ("OAT3", "SLC22A8", "organic anion transporter 3")),
    ("SLC47A1", "MATE1", "SLC47A1", "transporter", ("MATE1", "SLC47A1", "multidrug and toxin extrusion 1")),
    ("SLC47A2", "MATE2-K", "SLC47A2", "transporter", ("MATE2-K", "MATE2K", "SLC47A2")),
    ("ABCB11", "Bile salt export pump", "ABCB11", "transporter", ("BSEP", "ABCB11", "bile salt export pump")),
    ("SLC10A1", "NTCP", "SLC10A1", "transporter", ("NTCP", "SLC10A1")),
    ("ABCC2", "MRP2", "ABCC2", "transporter", ("MRP2", "ABCC2")),
    ("ABCC3", "MRP3", "ABCC3", "transporter", ("MRP3", "ABCC3")),
    ("ABCC4", "MRP4", "ABCC4", "transporter", ("MRP4", "ABCC4")),
)


def extract_graph_facts(evidence: dict[str, Any]) -> list[GraphFact]:
    if evidence.get("evidence_type") not in {"adme", "ddi"}:
        return []
    source_text = str(evidence.get("source_text") or "").strip()
    if not source_text:
        return []
    compound_aliases = _compound_aliases(evidence)
    if not compound_aliases:
        return []

    facts: dict[tuple[str, str], GraphFact] = {}
    for sentence in _sentences(source_text):
        for relation_text, compound_text in _relation_clauses(sentence, compound_aliases):
            for target in find_target_mentions(relation_text):
                for relationship_type, confidence in _relationship_types(
                    relation_text,
                    compound_text,
                    target.matched_text,
                    str(evidence.get("compound_role") or ""),
                ):
                    key = (target.key, relationship_type)
                    fact = GraphFact(
                        target=target,
                        relationship_type=relationship_type,
                        evidence_sentence=sentence.strip(),
                        confidence=confidence,
                    )
                    if key not in facts or facts[key].confidence < confidence:
                        facts[key] = fact
    return sorted(facts.values(), key=lambda fact: (fact.target.key, fact.relationship_type))


def find_target_mentions(text: str) -> list[TargetMention]:
    mentions: dict[str, TargetMention] = {}
    for key, preferred_name, gene_symbol, target_type, aliases in STATIC_TARGETS:
        match = _first_alias_match(text, aliases)
        if match is not None:
            mentions[key] = TargetMention(key, preferred_name, gene_symbol, target_type, match.group(0))

    for match in re.finditer(r"(?<![A-Za-z0-9])(?:CYP|Cyp)[ -]?[0-9][A-Za-z](?:[0-9]+)?(?![A-Za-z0-9])", text):
        symbol = re.sub(r"[ -]", "", match.group(0)).upper()
        mentions.setdefault(symbol, TargetMention(symbol, symbol, symbol if symbol[-1].isdigit() else None, "enzyme", match.group(0)))
    for match in re.finditer(r"(?<![A-Za-z0-9])(?:UGT|Ugt)[ -]?[0-9][A-Za-z](?:[0-9]+)?(?![A-Za-z0-9])", text):
        symbol = re.sub(r"[ -]", "", match.group(0)).upper()
        mentions.setdefault(symbol, TargetMention(symbol, symbol, symbol, "enzyme", match.group(0)))
    return sorted(mentions.values(), key=lambda mention: mention.key)


def _compound_aliases(evidence: dict[str, Any]) -> tuple[str, ...]:
    values = (
        evidence.get("attributed_compound_name"),
        evidence.get("compound_name"),
        evidence.get("compound"),
    )
    aliases = []
    for value in values:
        alias = str(value or "").strip()
        if len(alias) >= 3 and alias.lower() not in {item.lower() for item in aliases}:
            aliases.append(alias)
    return tuple(aliases)


def _sentences(text: str) -> Iterable[str]:
    for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", text):
        normalized = " ".join(sentence.split())
        if normalized:
            yield normalized


def _relation_clauses(sentence: str, compound_aliases: tuple[str, ...]) -> Iterable[tuple[str, str]]:
    active_compound: str | None = None
    for clause in re.split(r"\s+(?:and|but|whereas)\s+|\s*;\s*", sentence, flags=re.IGNORECASE):
        clause = clause.strip(" ,")
        if not clause:
            continue
        compound_match = _first_alias_match(clause, compound_aliases)
        if compound_match is not None:
            active_compound = compound_match.group(0)
            yield clause, active_compound
            continue
        if active_compound and re.match(r"^(?:is|was|were|acts?|serves?|can|may)\b", clause, flags=re.IGNORECASE):
            yield f"{active_compound} {clause}", active_compound


def _first_alias_match(text: str, aliases: Iterable[str]) -> re.Match[str] | None:
    ordered = sorted({str(alias).strip() for alias in aliases if str(alias).strip()}, key=len, reverse=True)
    if not ordered:
        return None
    pattern = r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(alias) for alias in ordered) + r")(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE)


def _relationship_types(sentence: str, compound: str, target: str, compound_role: str) -> list[tuple[str, float]]:
    compound_pattern = re.escape(compound)
    target_pattern = re.escape(target)
    patterns: tuple[tuple[str, tuple[str, ...], float], ...] = (
        ("substrate_of", (
            rf"{compound_pattern}.{{0,50}}?\bsubstrates?\s+(?:of|for)\s+{target_pattern}",
            rf"{compound_pattern}.{{0,50}}?\b{target_pattern}\s+(?:probe\s+|sensitive\s+|known\s+|typical\s+)?substrate\b",
            rf"\b{target_pattern}\s+(?:probe\s+|sensitive\s+|known\s+|typical\s+)?substrate.{{0,30}}?{compound_pattern}",
            rf"{compound_pattern}.{{0,45}}?\bmetaboli[sz]ed\s+(?:primarily\s+|mainly\s+)?by\s+{target_pattern}",
            rf"metabolism\s+of\s+{compound_pattern}.{{0,45}}?(?:mediated|cataly[sz]ed)\s+(?:primarily\s+)?by\s+{target_pattern}",
        ), 0.92),
        ("inhibits", (
            rf"{compound_pattern}.{{0,35}}?\b(?:inhibits?|(?:is|acts\s+as)\s+an?\s+inhibitor\s+of)\s+{target_pattern}",
            rf"inhibition\s+of\s+{target_pattern}.{{0,20}}?\s+by\s+{compound_pattern}",
            rf"{target_pattern}.{{0,15}}?\b(?:is|was)\s+inhibited\s+by\s+{compound_pattern}",
        ), 0.93),
        ("induces", (
            rf"{compound_pattern}.{{0,35}}?\b(?:induces?|(?:is|acts\s+as)\s+an?\s+inducer\s+of)\s+{target_pattern}",
            rf"induction\s+of\s+{target_pattern}.{{0,20}}?\s+by\s+{compound_pattern}",
            rf"{target_pattern}.{{0,15}}?\b(?:is|was)\s+induced\s+by\s+{compound_pattern}",
        ), 0.93),
        ("transported_by", (
            rf"{compound_pattern}.{{0,45}}?\b(?:transported|taken\s+up|effluxed).{{0,35}}?\s+by\s+{target_pattern}",
            rf"{target_pattern}.{{0,20}}?\b(?:transports?|mediates?\s+(?:the\s+)?(?:uptake|efflux)\s+of)\s+{compound_pattern}",
            rf"(?:uptake|efflux)\s+of\s+{compound_pattern}.{{0,30}}?(?:mediated|facilitated)\s+by\s+{target_pattern}",
        ), 0.92),
    )
    results = []
    for relationship_type, relation_patterns, confidence in patterns:
        match = next((re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in relation_patterns if re.search(pattern, sentence, flags=re.IGNORECASE)), None)
        if match is None:
            continue
        matched_relation = match.group(0)
        negated = bool(re.search(r"\b(?:not|no|neither|nor)\b", matched_relation, flags=re.IGNORECASE))
        if negated and relationship_type in {"substrate_of", "transported_by"}:
            results.append((f"not_{relationship_type}", confidence))
        elif not negated:
            results.append((relationship_type, confidence))
    return results
