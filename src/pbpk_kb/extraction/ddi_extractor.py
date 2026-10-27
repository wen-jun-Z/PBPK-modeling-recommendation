from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedDDIStatement:
    interaction_type: str
    enzyme_or_transporter: str | None
    auc_ratio: float | None
    cmax_ratio: float | None
    study_type: str
    population: str | None
    evidence_text: str
    # [FIX 2026-07-02 · claude-opus-4-8] every enzyme/transporter mentioned in the sentence, and whether the
    # sentence is target-ambiguous (>1 distinct target) so downstream/curation knows the attribution is a guess.
    all_targets: tuple[str, ...] = ()
    ambiguous_target: bool = False


# [FIX 2026-07-02 · claude-opus-4-8] expanded target coverage to include the pathways of the chosen compounds:
# CYP1A2 (caffeine), OCT/MATE (metformin), CYP2C8 (repaglinide), plus common UGT/BCRP/OATP1B3. CYP3A4 is listed
# before CYP3A; \bCYP3A\b does not match inside "CYP3A4" (word boundary), so the two do not double-count.
_TARGET_PATTERNS: list[tuple[str, str]] = [
    ("CYP1A2", r"\bCYP1A2\b|\bcytochrome P450 1A2\b|\bCYP\s?1A2\b"),
    ("CYP2B6", r"\bCYP2B6\b"),
    ("CYP2C8", r"\bCYP2C8\b"),
    ("CYP2C9", r"\bCYP2C9\b"),
    ("CYP2C19", r"\bCYP2C19\b"),
    ("CYP2D6", r"\bCYP2D6\b"),
    ("CYP2E1", r"\bCYP2E1\b"),
    ("CYP3A4", r"\bCYP3A4\b|\bcytochrome P450 3A4\b|\bCYP\s?3A4\b"),
    ("CYP3A", r"\bCYP3A\b"),
    ("P-glycoprotein", r"\bP-gp\b|\bP-glycoprotein\b|\bABCB1\b|\bMDR1\b"),
    ("BCRP", r"\bBCRP\b|\bABCG2\b"),
    ("OATP1B1", r"\bOATP1B1\b|\bSLCO1B1\b"),
    ("OATP1B3", r"\bOATP1B3\b|\bSLCO1B3\b"),
    ("OCT1", r"\bOCT1\b|\bSLC22A1\b"),
    ("OCT2", r"\bOCT2\b|\bSLC22A2\b"),
    ("OCTN2", r"\bOCTN2\b|\bSLC22A5\b"),
    ("MATE1", r"\bMATE1\b|\bSLC47A1\b"),
    ("MATE2", r"\bMATE2\b|\bMATE2-K\b|\bSLC47A2\b"),
    ("UGT1A1", r"\bUGT1A1\b"),
    ("UGT2B7", r"\bUGT2B7\b"),
]
_COMPILED_TARGETS: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pattern, flags=re.IGNORECASE)) for name, pattern in _TARGET_PATTERNS
]

_DDI_KEYWORDS = re.compile(
    r"\b(interaction|interactions|coadministered|co-administration|inhibitor|inhibitors|inhibits?|"
    r"inducer|inducers|induces?|substrate|substrates|metaboli[sz]ed by|increase[ds]?|decrease[ds]?|"
    r"avoid|contraindicated)\b",
    flags=re.IGNORECASE,
)

_LABEL_SOURCES = {"dailymed", "openfda_drug_label", "ema_epar", "drugs_fda"}
_LITERATURE_SOURCES = {"pubmed", "pmc_oa"}


def extract_ddi_statements(text: str | None, *, source_id: str | None = None) -> list[ExtractedDDIStatement]:
    if not text:
        return []
    study_type = _study_type_for(source_id)
    statements: list[ExtractedDDIStatement] = []
    seen: set[tuple[str, str | None, str]] = set()
    for sentence in _sentences(text):
        if not _DDI_KEYWORDS.search(sentence):
            continue
        targets = _find_targets(sentence)
        if not targets and not re.search(r"\bAUC\b|\bCmax\b", sentence, flags=re.IGNORECASE):
            continue
        distinct = tuple(dict.fromkeys(name for name, _, _ in targets))
        primary = targets[0] if targets else None
        target_name = primary[0] if primary else None
        # [FIX 2026-07-02 · claude-opus-4-8] role is decided by the wording *around the target*, not by the first
        # keyword anywhere in the sentence, so "X is a substrate of CYP3A4; avoid CYP3A4 inhibitors" is a substrate,
        # not an inhibitor, statement.
        role = _role_near(sentence, primary[1], primary[2]) if primary else _role_global(sentence)
        interaction_type = _label(target_name, role)
        auc_ratio = _ratio_near(sentence, "AUC")
        cmax_ratio = _ratio_near(sentence, "Cmax")
        key = (interaction_type, target_name, sentence.lower())
        if key in seen:
            continue
        seen.add(key)
        statements.append(
            ExtractedDDIStatement(
                interaction_type=interaction_type,
                enzyme_or_transporter=target_name,
                auc_ratio=auc_ratio,
                cmax_ratio=cmax_ratio,
                study_type=study_type,
                population=_population_hint(sentence),
                evidence_text=sentence,
                all_targets=distinct,
                ambiguous_target=len(distinct) > 1,
            )
        )
    return statements


def _study_type_for(source_id: str | None) -> str:
    if source_id in _LABEL_SOURCES:
        return "label_statement"
    if source_id in _LITERATURE_SOURCES:
        return "literature_statement"
    return "statement"


def _sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [part.strip() for part in parts if len(part.strip()) >= 20]


def _find_targets(sentence: str) -> list[tuple[str, int, int]]:
    found: list[tuple[str, int, int]] = []
    for name, pattern in _COMPILED_TARGETS:
        match = pattern.search(sentence)
        if match:
            found.append((name, match.start(), match.end()))
    found.sort(key=lambda item: item[1])
    return found


def _role_near(sentence: str, start: int, end: int) -> str:
    window = sentence[max(0, start - 45):end + 45].lower()
    return _role_global(window)


def _role_global(text: str) -> str:
    lowered = text.lower()
    if "substrate" in lowered or "metaboli" in lowered or "transported by" in lowered or "cleared by" in lowered:
        return "substrate"
    if "inhibit" in lowered:
        return "inhibitor"
    if "induce" in lowered or "inducer" in lowered:
        return "inducer"
    if "contraindicated" in lowered or "avoid" in lowered:
        return "avoid coadministration"
    return "interaction"


def _label(target: str | None, role: str) -> str:
    base = target or "drug"
    if role == "avoid coadministration":
        return f"{base} avoid coadministration candidate"
    if role == "interaction":
        return f"{base} interaction candidate"
    return f"{base} {role} interaction candidate"


def _ratio_near(sentence: str, marker: str) -> float | None:
    patterns = [
        rf"{marker}[^.]{{0,80}}?(?:increased|increase|decreased|decrease)[^.]{{0,40}}?(\d+(?:\.\d+)?)\s*[- ]?fold",
        rf"(\d+(?:\.\d+)?)\s*[- ]?fold[^.]{{0,60}}?{marker}",
        rf"{marker}[^.]{{0,80}}?ratio[^0-9]{{0,20}}(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    percent_pattern = rf"{marker}[^.]{{0,80}}?(?:increased|increase) by\s+(\d+(?:\.\d+)?)\s*%"
    if match := re.search(percent_pattern, sentence, flags=re.IGNORECASE):
        return 1.0 + float(match.group(1)) / 100.0
    percent_reverse_pattern = rf"(?:increased|increase)[^.]{{0,80}}?{marker}[^.]{{0,40}}?by\s+(\d+(?:\.\d+)?)\s*%"
    if match := re.search(percent_reverse_pattern, sentence, flags=re.IGNORECASE):
        return 1.0 + float(match.group(1)) / 100.0
    percent_decrease_pattern = rf"{marker}[^.]{{0,80}}?(?:decreased|decrease) by\s+(\d+(?:\.\d+)?)\s*%"
    if match := re.search(percent_decrease_pattern, sentence, flags=re.IGNORECASE):
        return max(0.0, 1.0 - float(match.group(1)) / 100.0)
    percent_decrease_reverse_pattern = rf"(?:decreased|decrease)[^.]{{0,80}}?{marker}[^.]{{0,40}}?by\s+(\d+(?:\.\d+)?)\s*%"
    if match := re.search(percent_decrease_reverse_pattern, sentence, flags=re.IGNORECASE):
        return max(0.0, 1.0 - float(match.group(1)) / 100.0)
    return None


def _population_hint(sentence: str) -> str | None:
    if re.search(r"\bhealthy (subjects|volunteers|adults)\b", sentence, flags=re.IGNORECASE):
        return "healthy subjects"
    if re.search(r"\bpatients\b", sentence, flags=re.IGNORECASE):
        return "patients"
    return None
