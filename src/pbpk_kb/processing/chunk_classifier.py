from __future__ import annotations

from hashlib import sha256
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


EVIDENCE_SIGNAL_RE = re.compile(
    r"\b("
    r"C\s*max|AUC(?:0-[a-z0-9]+|inf|last)?|T\s*max|half[- ]life|t1/2|clearance|"
    r"volume of distribution|bioavailability|protein binding|absorption|distribution|"
    r"metabolism|excretion|CYP(?:3A4|3A|2C9|2C19|2D6)?|UGT|P-gp|P-glycoprotein|"
    # [FIX 2026-07-02 · claude-opus-4-8] OCT/MATE require a digit (OCT1/MATE1...) so month "Oct" and word "mate" no longer signal evidence.
    r"OATP|BCRP|OCT[0-9]|OCTN[0-9]|MATE[0-9]|transporter|drug interaction|drug interactions|coadministration|"
    r"co-administered|coadministered|renal impairment|"
    r"hepatic impairment|pediatric|paediatric|geriatric|elderly|pregnancy|lactation|"
    r"PBPK|physiologically based pharmacokinetic|PK-Sim|MoBi|Simcyp|GastroPlus"
    r")\b",
    flags=re.IGNORECASE,
)

SEARCH_RESULT_RE = re.compile(
    r"\b("
    r"esearchresult|idlist|webenv|querykey|retstart|retmax|pubmed search|"
    r"search result|result count|next page token"
    r")\b",
    flags=re.IGNORECASE,
)

STRUCTURED_FIELD_SOURCES = {
    "pubchem",
    "rxnorm",
    "chembl",
    "drugcentral",
    "gtex",
    "hpa",
    "nhanes",
    "aact",
}

JSON_FREE_TEXT_PATH_TERMS = {
    "abstract",
    "abstracttext",
    "briefsummary",
    "brieftitle",
    "clinical_pharmacology",
    "clinicalpharmacology",
    "comment",
    "comments",
    "description",
    "descriptions",
    "detaileddescription",
    "drug_interactions",
    "druginteractions",
    "eligibilitycriteria",
    "hepatic_impairment",
    "hepaticimpairment",
    "indications_and_usage",
    "indicationsandusage",
    "interventiondescription",
    "measure",
    "officialtitle",
    "outcome_description",
    "outcomedescription",
    "pediatric_use",
    "pediatricuse",
    "pharmacokinetics",
    "pregnancy",
    "renal_impairment",
    "renalaimpairment",
    "renalimpairment",
    "summary",
    "text",
    "use_in_specific_populations",
    "useinspecificpopulations",
}

DAILYMED_LLM_SECTION_RE = re.compile(
    r"\b("
    r"clinical pharmacology|pharmacokinetics?|absorption|distribution|metabolism|excretion|"
    r"drug interactions?|specific populations?|renal impairment|hepatic impairment|"
    r"pediatric|paediatric|geriatric|pregnancy|lactation|pharmacogenomics?"
    r")\b",
    flags=re.IGNORECASE,
)

DAILYMED_LLM_HEADING_RE = re.compile(
    r"^\s*(?:[0-9]+(?:\.[0-9]+)?\s+)?("
    r"clinical pharmacology|pharmacokinetics?|absorption|distribution|metabolism|excretion|"
    r"drug interactions?|specific populations?|renal impairment|hepatic impairment|"
    r"pediatric|paediatric|geriatric|pregnancy|lactation|pharmacogenomics?"
    r")\b",
    flags=re.IGNORECASE,
)

DAILYMED_LOW_VALUE_SECTION_RE = re.compile(
    r"\b("
    r"warnings?|precautions?|dosage|indications?|contraindications?|boxed warning|"
    r"adverse reactions?|overdosage|how supplied|storage|patient counseling"
    r")\b",
    flags=re.IGNORECASE,
)

PUBMED_METADATA_RE = re.compile(
    r"\b("
    r"articleids|recordstatus|pubstatus|sortpubdate|sortfirstauthor|"
    r"fulljournalname|doctype|nlmuniqueid|pmcrefcount|vernaculartitle|"
    r"srccontriblist|pubtype|has abstract"
    r")\b",
    flags=re.IGNORECASE,
)

NAVIGATION_OR_FILTER_RE = re.compile(
    r"\b("
    r"search fields|downloadable data|tau score|category tissue|category cell type|"
    r"cluster [0-9]+:|contact news|privacy statement|licence|help & faq"
    r")\b",
    flags=re.IGNORECASE,
)

PBPK_MODEL_MARKER_RE = re.compile(
    r"\bpbpk\b|physiolog(?:ically|ical)[- ]based|whole[- ]body|\bpk[- ]?sim\b|\bsimcyp\b",
    flags=re.IGNORECASE,
)

NON_RESULT_SECTION_RE = re.compile(
    r"\b("
    r"references?|bibliograph(?:y|ies)|literature cited|endnote bibliography|"
    r"introduction|background|materials? and methods?|methods?|study design|"
    r"statistical analysis|statistical methods?|objectives?|aims?|eligibility\s*criteria|eligibility|"
    r"indications? and usage|dosage|dosing|warnings?|precautions?|"
    r"adverse reactions?|overdosage|patient counseling|clinical studies|"
    r"safety|efficacy|pharmacodynamics?"
    r")\b",
    flags=re.IGNORECASE,
)

OPENFDA_ALLOWED_FIELD_TERMS = {
    "clinicalpharmacology",
    "pharmacokinetics",
    "druginteractions",
    "useinspecificpopulations",
    "renalimpairment",
    "hepaticimpairment",
    "pediatricuse",
    "geriatricuse",
    "pregnancy",
    "lactation",
    "nursingmothers",
}

OUTCOME_PARAMETER_RE = re.compile(
    r"\b("
    r"C\s*max|AUC(?:0-[a-z0-9]+|inf|last)?|T\s*max|half[- ]life|t1/2|"
    r"clearance|volume of distribution|bioavailability|systemic availability|"
    r"protein bound|protein binding|fraction unbound|plasma concentration|"
    r"blood concentration|serum concentration|steady[- ]state concentration|"
    r"exposure|absorption|distribution|metabolism|excretion|elimination|"
    r"pharmacokinetics?|intrinsic clearance|renal clearance"
    r")\b",
    flags=re.IGNORECASE,
)

SUBJECT_RENAL_FUNCTION_RE = re.compile(
    r"\b(?:creatinine\s+clearance|CrCl|CL\s*_?\s*cr|eGFR|estimated\s+glomerular\s+filtration\s+rate)\b",
    flags=re.IGNORECASE,
)

RESULT_RELATION_RE = re.compile(
    r"\b("
    r"increas(?:e|ed|es|ing)|decreas(?:e|ed|es|ing)|reduc(?:e|ed|es|ing)|"
    r"prolong(?:ed|s|ing)?|shorten(?:ed|s|ing)?|higher|lower|greater|less|"
    r"unchanged|similar|different|dose[- ]proportional|non[- ]?linear|linear|"
    r"reached|ranged|averaged|geometric mean ratio|accounted for|resulted in|led to|was found|were found|"
    r"rapidly absorbed|poorly absorbed|completely absorbed"
    r")\b",
    flags=re.IGNORECASE,
)

MECHANISM_RELATION_RE = re.compile(
    r"\b("
    r"(?:is|was|are|were)\s+(?:primarily\s+|mainly\s+|predominantly\s+)?"
    r"(?:metabolized|metabolised|excreted|eliminated|transported|absorbed|bound)|"
    r"metabolized by|metabolised by|substrate of|substrate for|"
    r"inhibits?|induces?|mediated by|transported by|excreted unchanged|"
    r"eliminated unchanged|undergoes? metabolism|binds? to"
    r")\b",
    flags=re.IGNORECASE,
)

MECHANISM_TERM_RE = re.compile(
    r"\b(CYP[0-9A-Z]*|UGT[0-9A-Z]*|P-gp|P-glycoprotein|OATP[0-9A-Z]*|BCRP|"
    r"OCT[0-9]|OCTN[0-9]|MATE[0-9]|transporter|enzyme|metabolized|metabolised|"
    r"excreted|eliminated|absorbed|protein bound|substrate|inhibitor|inducer)\b",
    flags=re.IGNORECASE,
)

MODEL_RELATION_RE = re.compile(
    r"\b(developed|built|constructed|implemented|validated|evaluated|qualified|"
    r"predicted|simulated|reproduced|calibrated|fitted|applies|describes|allows|"
    r"based\s+on)\b",
    flags=re.IGNORECASE,
)

STUDY_INTENT_RE = re.compile(
    r"\b(?:purpose|objective|aim|designed)\b.{0,100}\b(?:determine|evaluate|assess|"
    r"investigate|measure|characterize|compare)|\bto\s+(?:determine|evaluate|assess|"
    r"investigate|measure|characterize|compare)\b|"
    r"\b(?:was|were|is|are|has\s+been|have\s+been)\s+"
    r"(?:determined|evaluated|assessed|investigated|measured|characterized|compared)\b",
    flags=re.IGNORECASE,
)

PARAMETER_ESTIMATION_METHOD_RE = re.compile(
    r"\b(?:regression|analysis|model|method)\b.{0,100}\b(?:to|for)\s+(?:estimate|predict|calculate)\b|"
    r"\b(?:estimate|predict|calculate)\b.{0,80}\b(?:by|using|with)\b.{0,40}\b(?:method|model|regression)\b",
    flags=re.IGNORECASE,
)

LOW_VALUE_SENTENCE_RE = re.compile(
    r"\b(?:sedation|respiratory depression|adverse events?|adverse reactions?|"
    r"efficacy|therapeutic response|indicated for|dose recommendations?|"
    r"monitor patients?|patient counseling|contraindicated)\b",
    flags=re.IGNORECASE,
)

CLINICAL_RECOMMENDATION_RE = re.compile(
    r"\b(?:should|must)\s+(?:not\s+)?be\s+(?:initiated|used|administered|given|prescribed)|"
    r"\b(?:dose|dosage)\s+(?:adjustment|reduction|increase|recommendation)|"
    r"\b(?:monitor|monitoring)\s+(?:the\s+)?patients?\b|\bcontraindicat(?:ed|ion)\b",
    flags=re.IGNORECASE,
)

STRONG_PK_PARAMETER_RE = re.compile(
    r"\b(?:C\s*max|AUC(?:0-[a-z0-9]+|inf|last)?|T\s*max|half[- ]life|t1/2|"
    r"clearance|volume of distribution|bioavailability|fraction unbound|protein binding|"
    r"plasma concentration|blood concentration|serum concentration|exposure)\b",
    flags=re.IGNORECASE,
)

PHARMACODYNAMIC_CONTEXT_RE = re.compile(
    r"\b(?:cAMP|cyclic AMP|EEG|electroencephalogram|sedation|respiratory depression|"
    r"blood pressure|heart rate|analgesia|pain|efficacy|tumou?r|cell proliferation|"
    r"intracellular|sperm|glucose lowering|insulin sensitiz|receptor activation)\b",
    flags=re.IGNORECASE,
)

NON_OBSERVED_ASSERTION_RE = re.compile(
    r"\b(?:has not been|have not been|was not|were not)\s+(?:formally\s+)?(?:studied|evaluated|determined)|"
    r"\bno\s+(?:formal\s+)?(?:pharmacokinetic\s+)?stud(?:y|ies)\b|"
    r"\bno\s+(?:clinically\s+significant\s+)?drug\s+interaction\b|"
    r"\bnot expected\b|\bunknown\b|\bnot known\b",
    flags=re.IGNORECASE,
)

INCOMPLETE_SENTENCE_END_RE = re.compile(
    r"(?:\b(?:from|to|between|and|or|by|with|of|than|versus|vs\.?|was|were|is|are|"
    r"in|after|before|during|following|when|while|whereas|because|if|unless|until|"
    r"with no|almost|nearly|approximately|about|Eq\.|with\s+(?:the\s+)?(?:higher|lower))|[,;:])\s*$",
    flags=re.IGNORECASE,
)

INCOMPLETE_SENTENCE_START_RE = re.compile(
    r"^\s*(?:(?:of|and|or|than|versus|vs\.?)\b|"
    r"(?:suggest(?:s|ed)?|indicat(?:e|es|ed))\s+that\b)",
    flags=re.IGNORECASE,
)

DANGLING_UNPUNCTUATED_START_RE = re.compile(
    r"^\s*(?:although|whereas|because|approximately|suggest(?:s|ed)?\s+that|"
    r"indicat(?:e|es|ed)\s+that|delays?\b)",
    flags=re.IGNORECASE,
)

DANGLING_UNPUNCTUATED_END_RE = re.compile(
    r"(?:\bin\s+conjunction|\bas\s+shown|\bdoes\s+not\s+undergo|"
    r"\b(?:similar\s+to\s+or|similar\s+or)\s+(?:higher|lower|longer|shorter)|"
    r"\b(?:reported|matched)\s+(?:for|with|to)\s+(?:single|the|a|an)|"
    r"\b(?:to|into)\s+(?:a|an|the)|"
    r"\b(?:and|or)\s+the(?:\s+[A-Za-z][A-Za-z-]*){0,3}|"
    r"\bwithout\s+(?:affecting|changing|altering)\s+the\s+[A-Za-z]|"
    r"\b(?:in\s+the\s+alcoholic|gender-and\s+weight-matched\s+healthy|cardiac\s+output\s+and\s+hepatic))\s*$",
    flags=re.IGNORECASE,
)

SAFE_UNPUNCTUATED_VALUE_END_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d+)?)(?:\s*(?:\+/-|±)\s*\d+(?:\.\d+)?)?\s*"
    r"(?:%|[- ]?fold|pg|ng|ug|µg|μg|mg|g|kg|pmol|nmol|umol|µmol|μmol|mmol|mol|"
    r"mL|L|sec(?:ond)?s?|min(?:ute)?s?|h|hr|hrs|hours?|days?|weeks?)"
    r"(?:\s*[/.]\s*[A-Za-z0-9µμ^()*-]+)*\s*$",
    flags=re.IGNORECASE,
)

SAFE_UNPUNCTUATED_PREDICATE_END_RE = re.compile(
    r"\b(?:decreased|increased|reduced|prolonged|shortened|unchanged|similar|higher|lower|"
    r"greater|different|absorbed|metabolized|metabolised|excreted|eliminated|transported|"
    r"bound|inhibited|induced)\s*$",
    flags=re.IGNORECASE,
)

SAFE_UNPUNCTUATED_RELATION_END_RE = re.compile(
    r"(?:\b(?:metabolized|metabolised|transported|mediated|inhibited|induced|substrate)\b"
    r".{0,80}\b(?:CYP[0-9A-Z]*|UGT[0-9A-Z]*|P[- ]?gp|OATP[0-9A-Z]*|BCRP|OCT[0-9]|MATE[0-9]|"
    r"cytochrome\s+P450(?:\s+[A-Z0-9]+)?)|"
    r"\b(?:excreted|eliminated)\s+unchanged\s+in\s+(?:the\s+)?urine|"
    r"\bdoes\s+not\s+undergo\s+(?:hepatic\s+)?metabolism|"
    r"\b(?:compared\s+(?:with|to)|than\s+in|observed\s+in)\s+"
    r"(?:healthy\s+)?(?:subjects?|volunteers?|adults?|children|patients?))\s*$",
    flags=re.IGNORECASE,
)

FLATTENED_TABLE_HEADER_RE = re.compile(
    r"\b(?:table\s+\d+|geometric\s+mean\s+ratio|ratio\s+of\s+arithmetic\s+means|"
    r"no\s+effect\s*=\s*1(?:\.0+)?|p-value\s+of\s+difference)\b",
    flags=re.IGNORECASE,
)

FLATTENED_TABLE_MARKER_RE = re.compile(r"[†‡§¶]")

# Flattened label tables may lose their heading and retain only lettered
# footnotes. Adjacent mean (±SD) cells have no parameter or column ownership;
# keep the complete mixed fragment for table recovery instead of extracting it.
_TABLE_SUMMARY_CELL_PATTERN = (
    r"[-+]?\d+(?:\.\d+)?\s*\(\s*(?:±|\+\s*/\s*[-−])\s*\d+(?:\.\d+)?\s*\)"
)
FLATTENED_TABLE_SUMMARY_CELLS_RE = re.compile(
    rf"(?<![\w.])(?:{_TABLE_SUMMARY_CELL_PATTERN}\s+){{2}}{_TABLE_SUMMARY_CELL_PATTERN}"
)

_TABLE_NUMBER_PATTERN = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
_TABLE_UNPAREN_SD_CELL_PATTERN = (
    rf"{_TABLE_NUMBER_PATTERN}\s*(?:±|\+\s*/\s*[-−])\s*{_TABLE_NUMBER_PATTERN}"
)
FLATTENED_TABLE_TWO_SD_COLUMNS_RE = re.compile(
    rf"(?<![\w.,]){_TABLE_UNPAREN_SD_CELL_PATTERN}\s+{_TABLE_UNPAREN_SD_CELL_PATTERN}(?![\w.,])"
)
FLATTENED_TABLE_PK_SD_HEADER_RE = re.compile(
    r"\bpharmacokinetic\s+parameters\s*\(\s*mean\s*(?:±|\+\s*/\s*[-−])\s*s\.?\s*d\.?\s*\)",
    flags=re.IGNORECASE,
)

# Unit-bearing parameter names directly beside each other are table columns,
# even when the table caption and the explicit mean/SD heading were lost.
_TABLE_PK_COLUMN_PATTERN = (
    r"\b(?:C\s*max|T\s*max|AUC|half[- ]life|clearance)"
    r"\s*\([a-zμµ][^()\n]{0,23}\)"
)
FLATTENED_TABLE_PK_COLUMNS_RE = re.compile(
    rf"{_TABLE_PK_COLUMN_PATTERN}\s+{_TABLE_PK_COLUMN_PATTERN}",
    flags=re.IGNORECASE,
)
FLATTENED_TABLE_RANGE_RE = re.compile(
    r"\(\s*[-+]?\d+(?:\.\d+)?\s*(?:[-–]|to)\s*[-+]?\d+(?:\.\d+)?\s*\)",
    flags=re.IGNORECASE,
)

REFERENCE_SENTENCE_RE = re.compile(
    r"^\s*(?:\[[0-9]+\]\s*)?(?:et\s+al\.|[A-Z][A-Za-z-]+\s+et\s+al\.)|"
    r"\b(?:doi:|vol\.|pp\.|journal of|references?)\b",
    flags=re.IGNORECASE,
)

UNCERTAIN_ASSERTION_RE = re.compile(
    r"\b(?:may|might|could|possibly|potentially|hypothes(?:is|ize|ized|izes)|"
    r"expected to|anticipated to|whether or not|probably|likely|predicted|predicts?)\b",
    flags=re.IGNORECASE,
)

OBSERVED_RESULT_RE = re.compile(
    r"\b(?:observed|found|showed|demonstrated|resulted in|was|were)\b.{0,45}"
    r"\b(?:increased|decreased|reduced|prolonged|shortened|higher|lower|unchanged|different)\b",
    flags=re.IGNORECASE,
)

NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d+)?)"
CONCENTRATION_PARAMETER_RE = re.compile(r"\b(?:C\s*max|plasma|blood|serum|steady[- ]state)\b.{0,35}\bconcentration\b|\bC\s*max\b", re.IGNORECASE)
CONCENTRATION_VALUE_RE = re.compile(rf"{NUMBER_PATTERN}\s*(?:ng/mL|µg/mL|ug/mL|mcg/mL|mg/L|µmol/L|umol/L)\b", re.IGNORECASE)
AUC_PARAMETER_RE = re.compile(r"\bAUC(?:0-[a-z0-9]+|inf|last)?\b|area under (?:the )?curve", re.IGNORECASE)
AUC_VALUE_RE = re.compile(
    rf"{NUMBER_PATTERN}\s*(?:ng\s*[·*x-]?\s*h\s*/\s*mL|µg\s*[·*x-]?\s*h\s*/\s*mL|"
    r"ug\s*[·*x-]?\s*h\s*/\s*mL|mg\s*[·*x-]?\s*h\s*/\s*L|%|fold)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
TIME_PARAMETER_RE = re.compile(r"\b(?:T\s*max|half[- ]life|t1/2|time to (?:peak|maximum))\b", re.IGNORECASE)
TIME_VALUE_RE = re.compile(rf"{NUMBER_PATTERN}\s*(?:h(?:ours?)?|hr|hrs|min(?:utes?)?|sec(?:onds?)?|days?)\b", re.IGNORECASE)
CLEARANCE_PARAMETER_RE = re.compile(r"\b(?:clearance|intrinsic clearance|renal clearance)\b", re.IGNORECASE)
CLEARANCE_VALUE_RE = re.compile(rf"{NUMBER_PATTERN}\s*(?:L/h(?:/kg)?|L/hr(?:/kg)?|mL/min(?:/kg)?|mL/h(?:/kg)?|%)(?![A-Za-z0-9])", re.IGNORECASE)
VOLUME_PARAMETER_RE = re.compile(r"\b(?:volume of distribution|Vd|Vss|V/F)\b", re.IGNORECASE)
VOLUME_VALUE_RE = re.compile(rf"{NUMBER_PATTERN}\s*(?:L|mL)(?:/kg)?\b", re.IGNORECASE)
FRACTION_PARAMETER_RE = re.compile(r"\b(?:bioavailability|systemic availability|protein bound|protein binding|fraction unbound)\b", re.IGNORECASE)
FRACTION_VALUE_RE = re.compile(rf"{NUMBER_PATTERN}\s*%", re.IGNORECASE)


@dataclass(frozen=True)
class ChunkDecision:
    status: str
    action: str
    reason: str
    llm_required: bool
    evidence_signal: bool
    natural_language_score: float
    machine_json_score: float


@dataclass(frozen=True)
class PreparedLLMInput:
    eligible: bool
    status: str
    reason: str
    sentences: tuple[str, ...] = ()
    sentence_fingerprints: tuple[str, ...] = ()

    @property
    def input_text(self) -> str:
        return "\n".join(self.sentences)


@dataclass(frozen=True)
class _SentenceSpan:
    start: int
    end: int
    text: str


def classify_chunk(chunk: dict[str, Any], *, min_chars: int, existing_evidence_count: int = 0) -> ChunkDecision:
    text = str(chunk.get("chunk_text") or "")
    source_id = str(chunk.get("source_id") or "")
    source_location = str(chunk.get("source_location") or "")
    length = len(text.strip())
    evidence_signal = bool(EVIDENCE_SIGNAL_RE.search(text))
    natural_score = _natural_language_score(text)
    machine_score = _machine_json_score(text)

    if length < min_chars:
        return ChunkDecision(
            status="skipped_too_short",
            action="skip",
            reason=f"text length {length} < min_chars {min_chars}",
            llm_required=False,
            evidence_signal=evidence_signal,
            natural_language_score=natural_score,
            machine_json_score=machine_score,
        )

    if _is_search_result_list(text):
        return ChunkDecision(
            status="skipped_search_result_list",
            action="skip",
            reason="search/list metadata is not direct scientific evidence",
            llm_required=False,
            evidence_signal=evidence_signal,
            natural_language_score=natural_score,
            machine_json_score=machine_score,
        )

    if _is_machine_structured_chunk(text, source_id=source_id, source_location=source_location, natural_score=natural_score, machine_score=machine_score):
        status = "structured_or_machine_chunk"
        action = "structured_mapping_or_skip"
        if existing_evidence_count:
            status = "structured_or_rule_extracted"
            action = "already_extracted_without_llm"
        return ChunkDecision(
            status=status,
            action=action,
            reason="structured/API machine payload is handled by field mapping or skipped as non-prose",
            llm_required=False,
            evidence_signal=evidence_signal,
            natural_language_score=natural_score,
            machine_json_score=machine_score,
        )

    source_allowed, source_reason = _source_allows_llm(chunk, text=text, source_id=source_id, source_location=source_location, machine_score=machine_score)
    if not source_allowed:
        return ChunkDecision(
            status="structured_or_machine_chunk",
            action="structured_mapping_or_skip",
            reason=source_reason,
            llm_required=False,
            evidence_signal=evidence_signal,
            natural_language_score=natural_score,
            machine_json_score=machine_score,
        )

    if not evidence_signal:
        status = "existing_extracted_no_llm" if existing_evidence_count else "skipped_no_evidence_terms"
        action = "already_extracted_without_llm" if existing_evidence_count else "skip"
        return ChunkDecision(
            status=status,
            action=action,
            reason="no PK/ADME/DDI/special-population/PBPK signal detected",
            llm_required=False,
            evidence_signal=False,
            natural_language_score=natural_score,
            machine_json_score=machine_score,
        )

    if natural_score < 0.18:
        return ChunkDecision(
            status="skipped_non_natural_language",
            action="skip",
            reason="evidence terms found but text is mostly identifiers, numbers, markup, or code-like payload",
            llm_required=False,
            evidence_signal=True,
            natural_language_score=natural_score,
            machine_json_score=machine_score,
        )

    return ChunkDecision(
        status="llm_required",
        action="send_to_llm",
        reason="natural-language or complex-table fragment contains evidence signals",
        llm_required=True,
        evidence_signal=True,
        natural_language_score=natural_score,
        machine_json_score=machine_score,
    )


def prepare_llm_input(chunk: dict[str, Any], target_compound: str) -> PreparedLLMInput:
    text = str(chunk.get("chunk_text") or "")
    target = target_compound.strip()
    if not text.strip() or not target:
        return PreparedLLMInput(False, "skipped_no_llm_input", "chunk text or target compound is empty")

    source_id = str(chunk.get("source_id") or "")
    section_title = _section_title(chunk)
    if NON_RESULT_SECTION_RE.search(section_title) and not (
        source_id == "osp_pbpk_model_library" and PBPK_MODEL_MARKER_RE.search(text)
    ):
        return PreparedLLMInput(
            False,
            "skipped_non_result_section",
            f"section is excluded from LLM evidence extraction: {section_title}",
        )

    is_flattened_table = _looks_like_flattened_table_row(text)
    if is_flattened_table and not chunk.get("parsed_table_id"):
        return PreparedLLMInput(
            False,
            "deferred_unparsed_flattened_table",
            "flattened table text has no parsed_table_id or reliable complete row boundary",
        )
    allow_flattened_table = bool(chunk.get("parsed_table_id"))
    spans = _sentence_spans(text)
    selected: list[str] = []
    for index, span in enumerate(spans):
        sentence = span.text
        if not _mentions_target(sentence, target):
            continue
        if _sentence_is_evidence_assertion(
            sentence,
            target,
            allow_flattened_table=allow_flattened_table,
        ):
            selected.append(sentence)
            continue
        if index + 1 >= len(spans) or not _sentence_is_evidence_context(
            sentence,
            target,
            allow_flattened_table=allow_flattened_table,
        ):
            continue
        following = spans[index + 1]
        if _mentions_target(following.text, target) and _sentence_is_evidence_assertion(
            following.text,
            target,
            allow_flattened_table=allow_flattened_table,
        ):
            continue
        if not _sentence_fragment_can_supply_result(
            following.text,
            allow_flattened_table=allow_flattened_table,
        ):
            continue
        combined = text[span.start : following.end].strip()
        if combined and _sentence_is_evidence_assertion(
            combined,
            target,
            allow_flattened_table=allow_flattened_table,
        ):
            selected.append(combined)

    unique_sentences: list[str] = []
    fingerprints: list[str] = []
    seen: set[str] = set()
    for sentence in selected:
        fingerprint = sentence_fingerprint(target, sentence)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_sentences.append(sentence)
        fingerprints.append(fingerprint)

    if not unique_sentences:
        return PreparedLLMInput(
            False,
            "skipped_no_complete_evidence_assertion",
            "no target-specific sentence contains a complete PK/ADME/DDI/special-population/PBPK assertion",
        )
    return PreparedLLMInput(
        True,
        "llm_required",
        "target-specific complete evidence assertion passed high-precision routing",
        tuple(unique_sentences),
        tuple(fingerprints),
    )


def sentence_fingerprint(target_compound: str, sentence: str) -> str:
    normalized_target = _normalize_for_dedup(target_compound)
    normalized_sentence = _normalize_for_dedup(sentence)
    return sha256(f"{normalized_target}\n{normalized_sentence}".encode("utf-8", errors="replace")).hexdigest()


def should_run_rule_extraction(chunk: dict[str, Any], *, min_chars: int = 40) -> bool:
    text = str(chunk.get("chunk_text") or "")
    source_id = str(chunk.get("source_id") or "")
    source_location = str(chunk.get("source_location") or "")
    if len(text.strip()) < min_chars:
        return False
    natural_score = _natural_language_score(text)
    machine_score = _machine_json_score(text)
    if _is_search_result_list(text):
        return False
    if _is_machine_structured_chunk(text, source_id=source_id, source_location=source_location, natural_score=natural_score, machine_score=machine_score):
        return False
    return natural_score >= 0.12


def _section_title(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata_json") if isinstance(chunk.get("metadata_json"), dict) else {}
    return str(metadata.get("section_title") or "").strip()


def _sentence_spans(text: str) -> list[_SentenceSpan]:
    spans: list[_SentenceSpan] = []
    start = 0
    for index, character in enumerate(text):
        boundary = character in "!?\n"
        if character == ".":
            boundary = _is_sentence_terminal_period(text, index)
        if not boundary:
            continue
        end = index + 1
        _append_sentence_span(spans, text, start, end)
        start = end
    _append_sentence_span(spans, text, start, len(text))
    return spans


def _append_sentence_span(spans: list[_SentenceSpan], text: str, start: int, end: int) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return
    sentence = text[start:end]
    if len(sentence) < 2 or re.fullmatch(r"\d+[.)]?", sentence):
        return
    spans.append(_SentenceSpan(start, end, sentence))


def _is_sentence_terminal_period(text: str, index: int) -> bool:
    if index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
        return False
    prefix = text[max(0, index - 14) : index + 1].casefold()
    if re.search(r"(?:\be\.g|\bi\.e|\bvs|\bet\s+al|\bfig|\bdr|\bmr|\bmrs|\bprof|\bs\.d|\bs\.e\.m|\bi\.v|\bp\.o)\.$", prefix):
        return False
    if unit_continues_after_period(text, index):
        return False
    return index + 1 >= len(text) or text[index + 1].isspace()


def unit_continues_after_period(text: str, index: int) -> bool:
    """A dot in L. h -1 or mg. kg -1 is multiplication, not punctuation."""
    prefix = text[max(0, index-12):index+1]
    suffix = text[index+1:index+40]
    return bool(re.search(r"\b(?:L|mL|ml|l|mg|ng|pg|ug|g|kg|h|hr|min)\.$", prefix)
                and re.match(r"\s*(?:kg|g|h|hr|min|day|L|mL|ml)\s*(?:[-−]\s*[1-9]|\^\s*-?[1-9]|/)", suffix))


def source_quote_unit_boundary_error(source_text: str, context: str) -> str | None:
    if not source_text.endswith('.'):
        return None
    starts = [match.start() for match in re.finditer(re.escape(source_text), context)]
    if starts and all(unit_continues_after_period(context, start+len(source_text)-1) for start in starts):
        return "source_quote_truncated_inside_compound_unit"
    return None


def _mentions_target(text: str, target_compound: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9]){re.escape(target_compound)}(?![A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    ) is not None


def _sentence_is_evidence_assertion(
    sentence: str,
    target_compound: str,
    *,
    allow_flattened_table: bool = False,
) -> bool:
    if len(sentence.strip()) < 35 or not EVIDENCE_SIGNAL_RE.search(sentence):
        return False
    if _is_incomplete_sentence(
        sentence,
        target_compound=target_compound,
        allow_flattened_table=allow_flattened_table,
    ) or REFERENCE_SENTENCE_RE.search(sentence):
        return False
    if NON_OBSERVED_ASSERTION_RE.search(sentence):
        return False
    if PHARMACODYNAMIC_CONTEXT_RE.search(sentence) and not STRONG_PK_PARAMETER_RE.search(sentence):
        return False
    if LOW_VALUE_SENTENCE_RE.search(sentence) and not OUTCOME_PARAMETER_RE.search(sentence):
        return False
    if CLINICAL_RECOMMENDATION_RE.search(sentence):
        return False
    has_explicit_pbpk_model = _target_pbpk_model_relation_is_local(sentence, target_compound)
    if PARAMETER_ESTIMATION_METHOD_RE.search(sentence) and not has_explicit_pbpk_model:
        return False
    if UNCERTAIN_ASSERTION_RE.search(sentence) and not has_explicit_pbpk_model:
        return False
    has_result = _sentence_has_result(sentence, target_compound)
    has_observed_numeric = _has_parameter_value_pair(sentence) and _target_parameter_are_local(sentence, target_compound)
    if STUDY_INTENT_RE.search(sentence) and not (has_observed_numeric or OBSERVED_RESULT_RE.search(sentence)):
        return False
    return has_result


def _sentence_has_result(sentence: str, target_compound: str | None = None) -> bool:
    outcome_sentence = _without_subject_renal_function(sentence)
    has_parameter = OUTCOME_PARAMETER_RE.search(outcome_sentence) is not None
    has_numeric_outcome = has_parameter and _has_parameter_value_pair(outcome_sentence)
    has_result_relation = has_parameter and _patterns_are_local(
        outcome_sentence, OUTCOME_PARAMETER_RE, RESULT_RELATION_RE, max_gap=80
    )
    has_mechanism = MECHANISM_TERM_RE.search(sentence) is not None and MECHANISM_RELATION_RE.search(sentence) is not None
    has_pbpk_model = _patterns_are_local(
        sentence, PBPK_MODEL_MARKER_RE, MODEL_RELATION_RE, max_gap=120
    )
    if target_compound:
        target_parameter_local = _target_parameter_are_local(sentence, target_compound)
        has_numeric_outcome = has_numeric_outcome and target_parameter_local
        has_result_relation = has_result_relation and target_parameter_local
        if _parameter_is_explicitly_owned_by_other(sentence, target_compound) and not _target_has_explicit_ddi_role(
            sentence, target_compound
        ):
            has_numeric_outcome = False
            has_result_relation = False
        has_mechanism = has_mechanism and _target_mechanism_is_local(sentence, target_compound)
        has_pbpk_model = _target_pbpk_model_relation_is_local(sentence, target_compound)
    return has_numeric_outcome or has_result_relation or has_mechanism or has_pbpk_model


def _sentence_fragment_can_supply_result(
    sentence: str,
    *,
    allow_flattened_table: bool = False,
) -> bool:
    if (
        _is_incomplete_sentence(sentence, allow_flattened_table=allow_flattened_table)
        or REFERENCE_SENTENCE_RE.search(sentence)
        or NON_OBSERVED_ASSERTION_RE.search(sentence)
        or UNCERTAIN_ASSERTION_RE.search(sentence)
        or CLINICAL_RECOMMENDATION_RE.search(sentence)
        or (PARAMETER_ESTIMATION_METHOD_RE.search(sentence) and not PBPK_MODEL_MARKER_RE.search(sentence))
    ):
        return False
    if PHARMACODYNAMIC_CONTEXT_RE.search(sentence) and not STRONG_PK_PARAMETER_RE.search(sentence):
        return False
    return _sentence_has_result(sentence)


def _sentence_is_evidence_context(
    sentence: str,
    target_compound: str,
    *,
    allow_flattened_table: bool = False,
) -> bool:
    if not _mentions_target(sentence, target_compound):
        return False
    if (
        _is_incomplete_sentence(
            sentence,
            target_compound=target_compound,
            allow_flattened_table=allow_flattened_table,
        )
        or REFERENCE_SENTENCE_RE.search(sentence)
        or NON_OBSERVED_ASSERTION_RE.search(sentence)
        or UNCERTAIN_ASSERTION_RE.search(sentence)
        or CLINICAL_RECOMMENDATION_RE.search(sentence)
    ):
        return False
    outcome_sentence = _without_subject_renal_function(sentence)
    if STUDY_INTENT_RE.search(sentence):
        return OUTCOME_PARAMETER_RE.search(outcome_sentence) is not None or re.search(r"\bpharmacokinetics?\b", sentence, re.IGNORECASE) is not None
    return bool(
        OUTCOME_PARAMETER_RE.search(outcome_sentence)
        or re.search(r"\b(?:parameters?|values?|results?|disposition)\b", sentence, re.IGNORECASE)
    )


def _has_parameter_value_pair(sentence: str) -> bool:
    sentence = _without_subject_renal_function(sentence)
    pairs = (
        (CONCENTRATION_PARAMETER_RE, CONCENTRATION_VALUE_RE),
        (AUC_PARAMETER_RE, AUC_VALUE_RE),
        (TIME_PARAMETER_RE, TIME_VALUE_RE),
        (CLEARANCE_PARAMETER_RE, CLEARANCE_VALUE_RE),
        (VOLUME_PARAMETER_RE, VOLUME_VALUE_RE),
        (FRACTION_PARAMETER_RE, FRACTION_VALUE_RE),
    )
    return any(_patterns_are_local(sentence, parameter, value, max_gap=100) for parameter, value in pairs)


def _target_parameter_are_local(sentence: str, target_compound: str) -> bool:
    sentence = _without_subject_renal_function(sentence)
    target_re = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(target_compound)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return _patterns_are_local(sentence, target_re, OUTCOME_PARAMETER_RE, max_gap=140)


def _target_mechanism_is_local(sentence: str, target_compound: str) -> bool:
    target = re.escape(target_compound)
    mechanism = (
        r"(?:metabolized|metabolised|excreted|eliminated|transported|absorbed|bound|"
        r"substrate|inhibitor|inducer|inhibits|induces)"
    )
    patterns = (
        rf"(?<![A-Za-z0-9]){target}(?![A-Za-z0-9]).{{0,80}}\b{mechanism}\b",
        rf"\b(?:metabolism|metabolization|excretion|elimination|absorption|transport|binding)\s+of\s+{target}\b",
        rf"\b(?:metabolizes|metabolises|transports|binds)\s+{target}\b",
    )
    return any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in patterns)


def _target_pbpk_model_relation_is_local(sentence: str, target_compound: str) -> bool:
    target_re = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(target_compound)}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return _patterns_are_local(sentence, target_re, PBPK_MODEL_MARKER_RE, max_gap=120) and _patterns_are_local(
        sentence, PBPK_MODEL_MARKER_RE, MODEL_RELATION_RE, max_gap=120
    )


def _parameter_is_explicitly_owned_by_other(sentence: str, target_compound: str) -> bool:
    target = target_compound.casefold()
    ignored_owners = {"the", "this", "that", "drug", "compound", "plasma", "blood", "serum",
                      "adults", "children", "infants", "patients", "subjects", "volunteers", "controls"}
    for parameter in OUTCOME_PARAMETER_RE.finditer(sentence):
        tail = sentence[parameter.end() : parameter.end() + 80]
        owner_match = re.match(
            r"\s+(?:(?:rates?|parameters?|values?)\s+)?(?:of|for)\s+(?:(?:the|plasma|blood|serum)\s+)*"
            r"([A-Za-z0-9][A-Za-z0-9'′_-]{2,})",
            tail,
            re.IGNORECASE,
        )
        if not owner_match:
            continue
        owner = owner_match.group(1).casefold()
        if owner in ignored_owners or owner in target or target in owner:
            continue
        return True
    return False


def _target_has_explicit_ddi_role(sentence: str, target_compound: str) -> bool:
    target = re.escape(target_compound)
    patterns = (
        rf"co[- ]?administration\s+with\s+{target}",
        rf"co[- ]?administered\s+with\s+{target}",
        rf"(?:presence|addition)\s+of\s+{target}",
        rf"\beffect\s+of\s+{target}\s+on\b",
        rf"\b(?:increased|decreased|reduced|prolonged|shortened)\b.{{0,100}}\bwith\s+{target}\b",
    )
    return any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in patterns)


def _patterns_are_local(
    text: str,
    first: re.Pattern[str],
    second: re.Pattern[str],
    *,
    max_gap: int,
) -> bool:
    first_matches = list(first.finditer(text))
    second_matches = list(second.finditer(text))
    for left in first_matches:
        for right in second_matches:
            if left.end() <= right.start():
                gap = right.start() - left.end()
            elif right.end() <= left.start():
                gap = left.start() - right.end()
            else:
                gap = 0
            if gap <= max_gap:
                return True
    return False


def _leading_assertion_context_error(source_text: str, target_compound: str | None) -> str | None:
    # A later sentence cannot supply the missing predicate or parameter owner
    # of a fragment at the start of the source quote.
    spans = _sentence_spans(source_text)
    first = spans[0].text if spans else source_text
    predicate_re = re.compile(
        r"\b(?:is|are|was|were|has|have|had)\b|" + RESULT_RELATION_RE.pattern,
        flags=re.IGNORECASE,
    )
    predicate = predicate_re.search(first)
    if (re.match(r"^(?:in|for|with|at|to|under|among)\b", first, re.IGNORECASE)
            and re.search(r"\brespectively[.!?]?\s*$", first, re.IGNORECASE)
            and OUTCOME_PARAMETER_RE.search(first) and predicate is None):
        return "prepositional_result_fragment_without_predicate"
    parameter = STRONG_PK_PARAMETER_RE.match(first)
    if parameter and predicate and target_compound:
        subject = first[:predicate.start()]
        # Explicit ownership can name another drug in a valid interaction
        # statement; attribution checks decide that role separately.
        owner = re.search(r"\b(?:of|for|to)\s+[A-Za-z][A-Za-z0-9_-]*", subject[parameter.end():], re.IGNORECASE)
        if not _mentions_target(subject, target_compound) and not owner:
            return "leading_pk_parameter_without_subject"
    return None


def source_assertion_completeness_error(
    source_text: str,
    *,
    target_compound: str | None = None,
    allow_flattened_table: bool = False,
) -> str | None:
    stripped = source_text.strip()
    if not stripped:
        return "empty_source_text"
    if re.search(r"\b(?:ABSTRACT|TEXT)\s+TRUNCATED\s+AT\s+\d+\s+WORDS\b", stripped, re.IGNORECASE):
        return "explicit_source_truncation_notice"
    if INCOMPLETE_SENTENCE_START_RE.search(stripped) or INCOMPLETE_SENTENCE_END_RE.search(stripped):
        return "dangling_sentence_boundary"
    if stripped.count("(") != stripped.count(")") or stripped.count("[") != stripped.count("]"):
        return "unbalanced_brackets"
    if not allow_flattened_table:
        context_error = _leading_assertion_context_error(stripped, target_compound)
        if context_error:
            return context_error
    if stripped[-1] in ".!?)]}":
        return None
    if DANGLING_UNPUNCTUATED_START_RE.search(stripped) or DANGLING_UNPUNCTUATED_END_RE.search(stripped):
        return "unfinished_natural_language_clause"
    if target_compound:
        target = re.escape(target_compound.strip())
        unfinished_target_clause = re.compile(
            rf"\b(?:when|while)\s+(?:the\s+)?{target}(?:\s+(?:was|were|is|are|has|had|been|"
            rf"being|taken|given|administered|co[- ]?administered)){{0,5}}(?:\s+in\s+conjunction)?\s*$",
            flags=re.IGNORECASE,
        )
        if unfinished_target_clause.search(stripped):
            return "unfinished_target_condition_clause"
    if allow_flattened_table:
        return None
    if SAFE_UNPUNCTUATED_VALUE_END_RE.search(stripped):
        return None
    if SAFE_UNPUNCTUATED_PREDICATE_END_RE.search(stripped):
        return None
    if SAFE_UNPUNCTUATED_RELATION_END_RE.search(stripped):
        return None
    return "unpunctuated_natural_language_fragment"


def _is_incomplete_sentence(
    sentence: str,
    *,
    target_compound: str | None = None,
    allow_flattened_table: bool = False,
) -> bool:
    return source_assertion_completeness_error(
        sentence,
        target_compound=target_compound,
        allow_flattened_table=allow_flattened_table,
    ) is not None


def _looks_like_flattened_table_row(text: str) -> bool:
    number_count = len(re.findall(NUMBER_PATTERN, text))
    marker_count = len(FLATTENED_TABLE_MARKER_RE.findall(text))
    range_count = len(FLATTENED_TABLE_RANGE_RE.findall(text))
    has_header = FLATTENED_TABLE_HEADER_RE.search(text) is not None
    has_adjacent_summary_cells = FLATTENED_TABLE_SUMMARY_CELLS_RE.search(text) is not None
    # Two group means can be normal prose. Require a PK matrix heading or
    # adjacent unit-bearing column labels, plus uninterrupted numeric cells.
    has_two_column_sd_matrix = bool(
        (FLATTENED_TABLE_PK_SD_HEADER_RE.search(text)
         or FLATTENED_TABLE_PK_COLUMNS_RE.search(text))
        and FLATTENED_TABLE_TWO_SD_COLUMNS_RE.search(text)
    )
    return (
        (has_header and number_count >= 3)
        or marker_count >= 2
        or (number_count >= 18 and range_count >= 5)
        or has_adjacent_summary_cells
        or has_two_column_sd_matrix
    )


def _without_subject_renal_function(sentence: str) -> str:
    return SUBJECT_RENAL_FUNCTION_RE.sub("subject_renal_function", sentence)


def _normalize_for_dedup(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("−", "-").replace("–", "-").replace("—", "-")
    normalized = re.sub(r"\[(?:\d+[\s,;-]*)+\]", " ", normalized)
    normalized = re.sub(r"[.!?]+\s*$", "", normalized)
    normalized = re.sub(r"[^a-z0-9%+./-]+", " ", normalized)
    return " ".join(normalized.split())


def _is_search_result_list(text: str) -> bool:
    lowered = text.lower()
    if SEARCH_RESULT_RE.search(text):
        return True
    pmid_count = len(re.findall(r"\b[1-9][0-9]{6,8}\b", text))
    title_like_count = len(re.findall(r"\b(title|pmid|pubmed|article ids?)\b", lowered))
    return pmid_count >= 8 and title_like_count >= 1 and _natural_language_score(text) < 0.22


def _is_machine_structured_chunk(text: str, *, source_id: str, source_location: str, natural_score: float, machine_score: float) -> bool:
    stripped = text.lstrip()
    if "#json_path=$#chunk=" in source_location:
        return True
    if _looks_like_xml_markup(text):
        return True
    if source_id == "pubmed" and "#xml_path=/#chunk=" in source_location and _is_pubmed_citation_metadata(text, machine_score=machine_score):
        return True
    if "#css=body#chunk=" in source_location and _looks_like_navigation_or_filter_page(text):
        return True
    if source_id in STRUCTURED_FIELD_SOURCES and "#json_path=" in source_location and machine_score >= 0.34 and natural_score < 0.3:
        return True
    if stripped.startswith(("{", "[")) and machine_score >= 0.42 and natural_score < 0.24:
        return True
    if _looks_like_numeric_array(text):
        return True
    if "PC_Compounds" in text and any(marker in text for marker in ['"atoms"', '"bonds"', '"coords"', '"conformers"']):
        return True
    return False


def _source_allows_llm(chunk: dict[str, Any], *, text: str, source_id: str, source_location: str, machine_score: float) -> tuple[bool, str]:
    if source_id in {"aact", "drugcentral", "hpa", "nhanes", "osp_tools_docs"}:
        return False, "structured registry/download source is not a direct LLM evidence chunk"
    section_title = _section_title(chunk)
    if NON_RESULT_SECTION_RE.search(section_title) and not (
        source_id == "osp_pbpk_model_library" and PBPK_MODEL_MARKER_RE.search(text)
    ):
        return False, "section is excluded because it does not contain evidence results"
    if "#json_path=" in source_location and not _json_path_is_allowed_free_text(source_location):
        return False, "JSON/API chunk is not from an approved free-text field"
    if source_id in {"clinicaltrials", "pubmed", "openfda_drug_label", "pkdb", "pmc_oa", "ema_epar"} and not _chunk_mentions_target_compound(chunk, text):
        return False, "compound-query source chunk does not mention the target compound in this text fragment"
    if source_id == "clinicaltrials" and not re.search(
        r"#json_path=.*(?:results?|outcomeMeasures?|outcomeAnalyses?|resultGroups?)",
        source_location,
        flags=re.IGNORECASE,
    ):
        return False, "ClinicalTrials descriptive protocol text is metadata, not posted PK evidence results"
    if source_id == "dailymed" and not _dailymed_chunk_is_allowed(chunk, text):
        return False, "DailyMed chunk is outside label sections/signals approved for LLM extraction"
    if source_id == "openfda_drug_label" and not _openfda_chunk_is_allowed(chunk, source_location):
        return False, "openFDA chunk is outside approved clinical label fields"
    if source_id == "pubmed" and _is_pubmed_citation_metadata(text, machine_score=machine_score):
        return False, "PubMed citation/search metadata is not direct scientific evidence"
    if source_id == "osp_pbpk_model_library" and not PBPK_MODEL_MARKER_RE.search(text):
        return False, "OSP text without an explicit PBPK model marker is not routed to LLM"
    if _looks_like_navigation_or_filter_page(text):
        return False, "navigation/filter/download page text is not direct scientific evidence"
    return True, "source and field are eligible for LLM extraction"


def _json_path_is_allowed_free_text(source_location: str) -> bool:
    path = source_location.split("#json_path=", 1)[-1].split("#chunk=", 1)[0]
    normalized = re.sub(r"[^a-z0-9_]+", "", path.lower())
    if normalized in {"", "$"}:
        return False
    return any(term in normalized for term in JSON_FREE_TEXT_PATH_TERMS)


def _chunk_mentions_target_compound(chunk: dict[str, Any], text: str) -> bool:
    compound = str(chunk.get("compound") or "").strip()
    if not compound or compound.lower() == "shared":
        return False
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(compound)}(?![A-Za-z0-9])", text, flags=re.IGNORECASE) is not None


def _dailymed_chunk_is_allowed(chunk: dict[str, Any], text: str) -> bool:
    metadata = chunk.get("metadata_json") if isinstance(chunk.get("metadata_json"), dict) else {}
    section_title = str(metadata.get("section_title") or "")
    header = " ".join(text.split()[:32])
    has_section_title = bool(section_title.strip())
    if not has_section_title:
        return DAILYMED_LLM_HEADING_RE.search(header) is not None
    if DAILYMED_LOW_VALUE_SECTION_RE.search(section_title):
        return False
    return DAILYMED_LLM_SECTION_RE.search(section_title) is not None


def _openfda_chunk_is_allowed(chunk: dict[str, Any], source_location: str) -> bool:
    metadata = chunk.get("metadata_json") if isinstance(chunk.get("metadata_json"), dict) else {}
    candidates = [str(metadata.get("section_title") or "")]
    if "#json_path=" in source_location:
        candidates.append(source_location.split("#json_path=", 1)[-1].split("#chunk=", 1)[0])
    normalized = {re.sub(r"[^a-z0-9]+", "", candidate.casefold()) for candidate in candidates if candidate}
    return any(any(term in candidate for term in OPENFDA_ALLOWED_FIELD_TERMS) for candidate in normalized)


def _is_pubmed_citation_metadata(text: str, *, machine_score: float) -> bool:
    return PUBMED_METADATA_RE.search(text) is not None and machine_score >= 0.12


def _looks_like_navigation_or_filter_page(text: str) -> bool:
    if NAVIGATION_OR_FILTER_RE.search(text):
        return True
    download_count = len(re.findall(r"\bdownload:\s*[^ ]+", text, flags=re.IGNORECASE))
    category_count = len(re.findall(r"\bcategory\b", text, flags=re.IGNORECASE))
    return download_count >= 3 and category_count >= 3


def _looks_like_xml_markup(text: str) -> bool:
    tag_count = len(re.findall(r"</?[A-Za-z][A-Za-z0-9:_-]*(?:\s[^>]*)?>", text))
    if tag_count < 8:
        return False
    tag_chars = sum(len(match.group(0)) for match in re.finditer(r"</?[A-Za-z][A-Za-z0-9:_-]*(?:\s[^>]*)?>", text))
    return tag_chars / max(1, len(text)) > 0.08


def _looks_like_numeric_array(text: str) -> bool:
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", text)
    if len(numbers) < 25:
        return False
    return len(numbers) > max(10, len(words) * 4)


def _natural_language_score(text: str) -> float:
    if not text:
        return 0.0
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", text)
    if not words:
        return 0.0
    common = re.findall(
        r"\b(the|and|or|of|in|to|with|for|was|were|is|are|patients?|subjects?|"
        r"dose|study|plasma|concentration|after|before|received|administered)\b",
        text,
        flags=re.IGNORECASE,
    )
    sentence_marks = len(re.findall(r"[.;:]\s+[A-Z]", text))
    return min(1.0, (len(words) / max(1, len(text) / 6.0)) * 0.55 + (len(common) / max(1, len(words))) * 0.35 + min(sentence_marks, 4) * 0.025)


def _machine_json_score(text: str) -> float:
    if not text:
        return 0.0
    machine_chars = sum(text.count(char) for char in ['{', '}', '[', ']', '"', ':', ','])
    return machine_chars / max(1, len(text))
