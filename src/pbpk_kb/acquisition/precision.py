"""High-precision candidate selection for public PBPK raw acquisition.

This module makes download decisions from source metadata only.  It never
extracts scientific values or creates evidence records.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any


HIGH_PRECISION_QUERY_FAMILIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "clinical_pk",
        "Clinical PK",
        (
            "pharmacokinetic",
            "pharmacokinetics",
            "AUC",
            "Cmax",
            "Tmax",
            "clearance",
            "half-life",
            "bioavailability",
            "concentration-time",
            "volume of distribution",
        ),
    ),
    (
        "adme",
        "ADME",
        (
            "absorption",
            "distribution",
            "protein binding",
            "fraction unbound",
            "metabolism",
            "excretion",
            "permeability",
            "CYP",
            "UGT",
            "transporter",
        ),
    ),
    (
        "ddi",
        "Drug-drug interaction",
        (
            "drug interaction",
            "drug-drug interaction",
            "coadministration",
            "inhibitor",
            "inducer",
            "victim drug",
            "perpetrator drug",
            "exposure",
            "AUC",
            "Cmax",
        ),
    ),
    (
        "special_population",
        "Special population PK",
        (
            "renal impairment",
            "hepatic impairment",
            "pediatric",
            "neonatal",
            "elderly",
            "pregnancy",
            "lactation",
            "pharmacogenetic",
            "pharmacogenomic",
            "obesity",
        ),
    ),
    (
        "pbpk_model",
        "PBPK model",
        (
            "physiologically based pharmacokinetic",
            "PBPK",
            "PK-Sim",
            "MoBi",
            "Simcyp",
            "GastroPlus",
            "population pharmacokinetic model",
        ),
    ),
    (
        "formulation_food_effect",
        "Formulation and food effect PK",
        (
            "food effect",
            "formulation",
            "dissolution",
            "permeability",
            "bioavailability",
            "modified release",
        ),
    ),
)

DIRECT_PK_TERMS = (
    "pharmacokinetic",
    "auc",
    "cmax",
    "tmax",
    "clearance",
    "half-life",
    "bioavailability",
    "concentration",
    "exposure",
    "protein binding",
    "fraction unbound",
    "metabolism",
    "excretion",
    "permeability",
    "drug interaction",
    "coadministration",
    "pbpk",
    "physiologically based pharmacokinetic",
)

# A full-text queue must use terms that explicitly describe PK, ADME, DDI, or
# PBPK work.  Generic "exposure" and "concentration" are useful discovery
# terms but are too broad to prove that the target compound is a study object.
STRONG_PBPK_TOPIC_TERMS = (
    "pharmacokinetic",
    "pharmacokinetics",
    "auc",
    "cmax",
    "tmax",
    "clearance",
    "half-life",
    "bioavailability",
    "volume of distribution",
    "protein binding",
    "fraction unbound",
    "absorption",
    "distribution",
    "metabolism",
    "excretion",
    "permeability",
    "transporter",
    "drug interaction",
    "drug-drug interaction",
    "coadministration",
    "inhibitor",
    "inducer",
    "victim drug",
    "perpetrator drug",
    "pbpk",
    "physiologically based pharmacokinetic",
    "pk-sim",
    "mobi",
    "simcyp",
    "gastroplus",
    "cyp3a",
    "cyp3a4",
    "cyp3a5",
)

# These terms narrow the ClinicalTrials.gov candidate API query before the
# stricter per-record intervention and PK-role check below.  They are discovery
# terms only; a record still needs the classifier's exact-target decision before
# its complete study JSON is downloaded.
CLINICALTRIAL_CANDIDATE_QUERY_TERMS = (
    "pharmacokinetic",
    "pharmacokinetics",
    "AUC",
    "Cmax",
    "Tmax",
    "clearance",
    "half-life",
    "bioavailability",
    "protein binding",
    "drug interaction",
    "coadministration",
    "PBPK",
    "renal impairment",
    "hepatic impairment",
    "pediatric",
    "pregnancy",
    "lactation",
    "food effect",
)

CLINICALTRIAL_CANDIDATE_FIELDS = "NCTId|BriefTitle|OfficialTitle|BriefSummary|InterventionName"

BACKGROUND_CONTEXT_TERMS = (
    "coffee enema",
    "caffeine enema",
    "caffeine enemas",
    "green tea",
    "dietary supplement",
    "fat burning",
    "food composition",
    "plant",
    "transgenic plant",
    "beverage consumption",
)


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def contains_exact_term(text: Any, term: str) -> bool:
    value = normalized_text(text)
    clean_term = normalized_text(term)
    if not value or not clean_term:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(clean_term)}(?![a-z0-9])", value) is not None


def matching_terms(text: Any, terms: tuple[str, ...] | list[str]) -> list[str]:
    return [term for term in terms if contains_exact_term(text, term)]


def high_precision_literature_query(
    compound: str,
    terms: tuple[str, ...] | list[str],
    *,
    family_id: str | None = None,
) -> str:
    """Build a high-precision PubMed candidate query.

    The target compound and source-topic terms must both occur in the title.
    Special-population and formulation queries additionally require a direct PK
    term in title/abstract, which avoids collecting general clinical-use papers.
    """

    compound_clause = f"{ncbi_fielded_term(compound, 'Title')}"
    topic_clause = " OR ".join(ncbi_fielded_term(term, "Title") for term in terms)
    query = f"({compound_clause}) AND ({topic_clause})"
    if family_id in {"special_population", "formulation_food_effect"}:
        pk_clause = " OR ".join(ncbi_fielded_term(term, "Title/Abstract") for term in STRONG_PBPK_TOPIC_TERMS)
        query += f" AND ({pk_clause})"
    return query


def clinicaltrial_candidate_query() -> str:
    """Build the public ClinicalTrials.gov topic prefilter for candidate discovery."""

    return " OR ".join(
        f'"{term}"' if " " in term else term for term in CLINICALTRIAL_CANDIDATE_QUERY_TERMS
    )


def ncbi_fielded_term(term: str, field: str) -> str:
    clean_term = str(term).strip().replace('"', r'\"')
    if " " in clean_term or "-" in clean_term:
        return f'"{clean_term}"[{field}]'
    return f"{clean_term}[{field}]"


def pubmed_records_from_xml(payload: bytes) -> list[dict[str, Any]]:
    """Return title/abstract metadata for every PubMedArticle in an EFetch payload."""

    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []
    records: list[dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = "".join(article.findtext(".//MedlineCitation/PMID", default="").split())
        if not pmid:
            continue
        title = " ".join("".join(article.find(".//ArticleTitle").itertext()).split()) if article.find(".//ArticleTitle") is not None else ""
        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            label = str(node.attrib.get("Label") or "").strip()
            value = " ".join("".join(node.itertext()).split())
            abstract_parts.append(f"{label}: {value}" if label and value else value)
        publication_types = [" ".join("".join(node.itertext()).split()) for node in article.findall(".//PublicationType")]
        records.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": " ".join(part for part in abstract_parts if part),
                "publication_types": publication_types,
            }
        )
    return records


def classify_pubmed_record(
    *,
    compound: str,
    title: str,
    abstract: str,
    query_families: set[str],
) -> dict[str, Any]:
    """Classify a literature record for raw-file download, not evidence quality."""

    text = " ".join(part for part in (title, abstract) if part)
    title_has_compound = contains_exact_term(title, compound)
    text_has_compound = contains_exact_term(text, compound)
    direct_terms = matching_terms(text, list(DIRECT_PK_TERMS))
    strong_terms = matching_terms(text, list(STRONG_PBPK_TOPIC_TERMS))
    strong_title_terms = matching_terms(title, list(STRONG_PBPK_TOPIC_TERMS))
    background_terms = matching_terms(text, list(BACKGROUND_CONTEXT_TERMS))
    family_terms = {
        family_id: matching_terms(text, terms)
        for family_id, _label, terms in HIGH_PRECISION_QUERY_FAMILIES
        if family_id in query_families
    }
    matched_family_terms = {family_id: terms for family_id, terms in family_terms.items() if terms}
    matched_title_family_terms = {
        family_id: matching_terms(title, terms)
        for family_id, _label, terms in HIGH_PRECISION_QUERY_FAMILIES
        if family_id in query_families and matching_terms(title, terms)
    }
    needs_abstract_pk_support = bool(query_families & {"special_population", "formulation_food_effect"})
    has_strong_title_or_supporting_pk_topic = bool(strong_title_terms) or (
        needs_abstract_pk_support and bool(strong_terms)
    )
    score = 0
    score += 5 if title_has_compound else 0
    score += min(4, len(strong_title_terms))
    score += min(2, len(matched_title_family_terms))
    score += 1 if needs_abstract_pk_support and bool(strong_terms) else 0
    if background_terms:
        score -= 5

    if not text_has_compound:
        decision = "excluded"
        reason = "target_compound_not_in_title_or_abstract"
    elif background_terms:
        decision = "metadata_only"
        reason = "background_or_food_context_requires_manual_relevance_review"
    elif title_has_compound and matched_title_family_terms and has_strong_title_or_supporting_pk_topic:
        decision = "fulltext_selected"
        reason = "exact_target_in_title_and_strong_pk_topic"
    elif not title_has_compound:
        decision = "metadata_only"
        reason = "target_not_in_title_requires_manual_role_review"
    elif not matched_title_family_terms:
        decision = "metadata_only"
        reason = "selected_topic_not_in_title_requires_manual_relevance_review"
    else:
        decision = "metadata_only"
        reason = "target_or_pk_topic_context_is_ambiguous"

    return {
        "decision": decision,
        "selection_reason": reason,
        "selection_score": score,
        "identity_match_type": (
            "exact_target" if decision == "fulltext_selected" else "mentioned_only" if text_has_compound else "other_component"
        ),
        "target_in_title": title_has_compound,
        "target_in_title_or_abstract": text_has_compound,
        "query_families": sorted(query_families),
        "matched_family_terms": matched_family_terms,
        "matched_title_family_terms": matched_title_family_terms,
        "matched_direct_pk_terms": direct_terms,
        "matched_strong_pbpk_terms": strong_terms,
        "matched_strong_pbpk_title_terms": strong_title_terms,
        "matched_background_terms": background_terms,
        "llm_triage_recommended": decision == "metadata_only" and text_has_compound and bool(matched_family_terms),
    }


def classify_dailymed_listing(
    *, compound: str, row: dict[str, Any], include_combination_labels: bool
) -> dict[str, Any]:
    title = str(row.get("title") or "")
    exact_target = contains_exact_term(title, compound)
    normalized_title = normalized_text(title)
    combination_markers = (" and ", " with ", " / ", "/", "+", ";")
    is_combination = any(marker in normalized_title for marker in combination_markers)
    if not exact_target:
        return {
            "decision": "excluded",
            "selection_reason": "target_not_in_spl_title",
            "identity_match_type": "other_component",
            "is_combination": is_combination,
        }
    if is_combination and not include_combination_labels:
        return {
            "decision": "metadata_only",
            "selection_reason": "combination_product_requires_component_specific_review",
            "identity_match_type": "target_component",
            "is_combination": True,
        }
    return {
        "decision": "fulltext_selected",
        "selection_reason": "exact_target_single_ingredient_spl_listing" if not is_combination else "combination_label_explicitly_enabled",
        "identity_match_type": "exact_target" if not is_combination else "target_component",
        "is_combination": is_combination,
    }


def classify_openfda_label(compound: str, label: dict[str, Any]) -> dict[str, Any]:
    """Classify an openFDA label result without interpreting label claims."""

    openfda = label.get("openfda") if isinstance(label, dict) else {}
    openfda = openfda if isinstance(openfda, dict) else {}
    names: list[str] = []
    for key in ("generic_name", "substance_name"):
        values = openfda.get(key) or []
        if isinstance(values, str):
            values = [values]
        names.extend(str(value) for value in values)
    product_text = " ".join(
        names
        + [
            str(value)
            for key in ("brand_name", "product_ndc")
            for value in (openfda.get(key) or [])
        ]
    )
    exact_target = any(contains_exact_term(name, compound) for name in names) or contains_exact_term(product_text, compound)
    normalized_names = normalized_text(" ".join(names))
    is_combination = any(marker in normalized_names for marker in (" and ", " with ", "/", "+", ";"))
    label_id = str(label.get("id") or label.get("set_id") or "")
    if not exact_target:
        return {
            "decision": "excluded",
            "selection_reason": "target_not_in_openfda_identity_fields",
            "identity_match_type": "other_component",
            "is_combination": is_combination,
            "label_id": label_id,
        }
    if is_combination:
        return {
            "decision": "metadata_only",
            "selection_reason": "combination_openfda_label_requires_component_specific_review",
            "identity_match_type": "target_component",
            "is_combination": True,
            "label_id": label_id,
        }
    return {
        "decision": "fulltext_selected",
        "selection_reason": "exact_target_openfda_label",
        "identity_match_type": "exact_target",
        "is_combination": False,
        "label_id": label_id,
    }


def clinical_trial_text(study: dict[str, Any]) -> tuple[str, list[str]]:
    protocol = study.get("protocolSection") if isinstance(study, dict) else {}
    protocol = protocol if isinstance(protocol, dict) else {}
    identification = protocol.get("identificationModule") or {}
    description = protocol.get("descriptionModule") or {}
    arms = protocol.get("armsInterventionsModule") or {}
    interventions: list[str] = []
    for intervention in arms.get("interventions") or []:
        if isinstance(intervention, dict):
            interventions.append(str(intervention.get("name") or ""))
            interventions.extend(str(item) for item in intervention.get("otherNames") or [])
    text = " ".join(
        [
            str(identification.get("briefTitle") or ""),
            str(identification.get("officialTitle") or ""),
            str(description.get("briefSummary") or ""),
            str(description.get("detailedDescription") or ""),
            " ".join(interventions),
        ]
    )
    return text, [item for item in interventions if item]


def clinical_trial_has_complete_protocol(study: dict[str, Any]) -> bool:
    """Return whether a ClinicalTrials record has more than field-limited candidate data."""

    protocol = study.get("protocolSection") if isinstance(study, dict) else None
    if not isinstance(protocol, dict) or not isinstance(protocol.get("identificationModule"), dict):
        return False
    return isinstance(protocol.get("statusModule"), dict) and any(
        isinstance(protocol.get(key), dict)
        for key in ("designModule", "armsInterventionsModule", "outcomesModule")
    )


def classify_clinical_trial(compound: str, study: dict[str, Any]) -> dict[str, Any]:
    text, interventions = clinical_trial_text(study)
    exact_intervention = any(contains_exact_term(value, compound) for value in interventions)
    exact_text = contains_exact_term(text, compound)
    pk_terms = matching_terms(text, list(DIRECT_PK_TERMS))
    if exact_intervention and pk_terms:
        return {
            "decision": "fulltext_selected",
            "selection_reason": "target_is_intervention_or_probe_with_pk_topic",
            "identity_match_type": "exact_target",
            "target_as_intervention": True,
            "matched_direct_pk_terms": pk_terms,
        }
    if exact_text and pk_terms:
        return {
            "decision": "metadata_only",
            "selection_reason": "target_or_role_requires_manual_review",
            "identity_match_type": "mentioned_only",
            "target_as_intervention": False,
            "matched_direct_pk_terms": pk_terms,
            "llm_triage_recommended": True,
        }
    return {
        "decision": "excluded",
        "selection_reason": "target_not_a_pk_relevant_intervention_or_probe",
        "identity_match_type": "mentioned_only",
        "target_as_intervention": False,
        "matched_direct_pk_terms": pk_terms,
    }


def decisions_counter(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("decision") or "unknown")] += 1
    return dict(sorted(counts.items()))
