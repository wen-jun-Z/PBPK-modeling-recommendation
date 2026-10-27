from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from copy import deepcopy
from hashlib import sha256
from typing import Any

from pbpk_kb.processing.evidence_llm import (DEFAULT_MODEL as DEFAULT_LLM_MODEL, LOCAL_BASE_URL as DEFAULT_LLM_BASE_URL, generate_evidence_json, validate_backend, backend_identity, server_backend, registered_server, generation_format)

LLM_EXTRACTION_PROMPT = """You are a PBPK public knowledge-base evidence extraction agent.

Task:
Extract only source-backed candidate evidence specifically about target_compound from the provided fragment.
The request contains input_items. Each item is one high-precision verbatim assertion made of one complete sentence or, when required, two adjacent context/result sentences. Evaluate each item independently. Never combine different items or non-adjacent statements.

Allowed evidence types:
- physicochemical
- clinical_pk
- adme
- ddi
- special_population
- tissue_expression
- public_pbpk_model

Evidence-domain boundaries:
- physicochemical: molecular weight, pKa, logP/logD, solubility, permeability, polar surface area, ionization, or formulation properties.
- adme: absorption, distribution, metabolism, excretion, bioavailability, fraction unbound, blood-to-plasma ratio, intrinsic/renal clearance, stability, permeability, efflux, enzyme/transporter substrate/inhibition/induction.
- clinical_pk: Cmax, AUC, Tmax, half-life, clearance, volume of distribution, renal clearance, accumulation, trough, concentration, or exposure. This is not pharmacodynamics, safety, efficacy, or dosing advice.
- ddi: target-compound exposure/PK change, exposure ratio, or explicit enzyme/transporter mechanism caused by another drug. This is not a generic warning about sedation or respiratory depression.
- special_population: a PK/ADME parameter or exposure difference in renal/hepatic impairment, pediatric, elderly, pregnancy, obesity, pharmacogenomic, alcoholic, or chronic-alcohol-use groups. This is not safety outcome or dose advice alone.
- tissue_expression: measured tissue abundance/expression of a PBPK-relevant protein, enzyme, or transporter.
- public_pbpk_model: public model identity, software, species, population, route, model file, parameter file, or evaluation metadata.

Rules:
1. Use only the supplied fragment. Do not use outside knowledge.
2. Return JSON only.
3. Return an object with key "candidates".
4. Each candidate must include:
   - target_compound
   - evidence_type
   - parameter
   - value
   - value_low
   - value_high
   - value_qualifier
   - unit
   - condition
   - source_text
   - confidence
   - requires_review
5. target_compound must exactly equal the target_compound in the request.
6. source_text must be copied exactly from one input_items item or table cell and must contain target_compound. When a result sentence omits the target or population named in its adjacent context sentence, source_text must include the complete contiguous two-sentence item, not the result sentence alone.
7. Extract only an assertion whose subject is target_compound. Do not assign evidence about a comparator, concomitant drug, excipient, metabolite, or mention-only drug to target_compound.
8. For DDI text, return evidence only when the relation and the role of target_compound are explicit. Put the interacting drug and direction in condition.
9. If no target-specific evidence is present, return {"candidates": []}.
10. Use value for one number. For a range use value=null plus numeric value_low and value_high. Never put a range string in value and never calculate a midpoint.
11. Do not infer missing values. For qualitative evidence use value=null, value_low=null, value_high=null, value_qualifier=null, and unit=null.
12. For quantitative evidence, copy every numeric value from source_text and copy its unit exactly. For an explicit mixed-unit time interval only, both endpoints may be converted into the same source-stated time unit; the validator checks both conversions against the original interval. Never use a dose or a different parameter\'s time values.
13. source_text must be a complete assertion. Do not return text cut off after words such as from, to, between, and, or, by, or in.
14. The parameter name must match terminology in source_text. Do not label steady-state concentration as Cmax, or a pharmacodynamic endpoint as a PK parameter.
15. If source_text directly states a quantitative change such as "56% reduction", return 56 with the copied percent-change unit instead of leaving value and unit null. Do not use dose or sample-size numbers as the result.
16. For group comparisons written as x vs y, x versus y, or paired values followed by respectively, do not assign a scalar or value_low/value_high range to a group. Return a qualitative candidate with all value fields, value_qualifier, and unit null, or return no candidate.
17. Capture explicit percent changes whether the direction follows the value (56% reduction) or precedes it (increased by approximately 60%).
18. Match unit dimensions to parameters: concentration for Cmax/concentration, AUC units for AUC, time for Tmax/half-life, clearance units for clearance, and volume units for volume.
19. Do not report dialysis/hemodialysis clearance or creatinine clearance/CrCl/eGFR as target-drug systemic clearance.
20. Do not emit tissue_expression. Stage 1 tissue expression is extracted deterministically from approved structured expression sources.
21. Emit public_pbpk_model only when source_text explicitly identifies PBPK, physiologically based or whole-body modeling, PK-Sim, or Simcyp. A population PK/PD or generic model is not PBPK.
22. Emit special_population only when source_text itself names the renal/hepatic impairment, pediatric, elderly, pregnancy, obesity, pharmacogenomic, or equivalent population. Do not rely on condition alone.
23. Do not turn search hit lists, entry pages, or failure records into evidence.
24. Do not classify pharmacodynamic endpoints, adverse effects, efficacy, safety outcomes, indications, or dose/administration instructions as PBPK evidence.
25. Mark every candidate requires_review=true.
26. Prefer one complete source sentence or a compact table fragment, not whole sections.
27. Reject a parameter jointly stated for target_compound and its metabolite, including hydroxy-, desmethyl-, or glucuronide metabolites. Extract only a separate parent-drug assertion.
28. For a unique bounded value such as less than 0.5%, set value=0.5, unit="%", and value_qualifier="<". Use "<=", ">", or ">=" when explicitly supported; otherwise value_qualifier=null.
29. For a unique explicit change such as increased 2.5-fold, set value=2.5 and unit="fold". Do not calculate a fold value from other numbers.
30. value, value_low and value_high MUST be JSON numbers or null. Model names, software names, mechanisms and other descriptive strings belong in condition, never in a value field.
31. Copy source_text character-for-character from an input_items element, including punctuation and formatting. Do not rewrite, concatenate bullet points, or repair source wording.
32. A range must have value=null, value_low=the lower endpoint, value_high=the upper endpoint. Never put one endpoint in value and the other in value_low.
33. Descriptions such as "inversely related" belong in condition, with all value fields, value_qualifier and unit null. Use parameter="excretion" for qualitative unchanged urinary excretion. value_qualifier is reserved for numerical bounds/ranges/confidence intervals.
34. For a comparison "7.88 h vs 5.37 h", return only a qualitative half_life assertion (all value fields and unit null). Do not generate one scalar candidate per group, even if group labels seem inferable.
35. A confidence interval such as "90% confidence interval: 87.9%-102%" has value=null, value_low=87.9, value_high=102, unit="%", value_qualifier="90% confidence interval". The hyphen between endpoints is not a negative sign; 90 is the confidence level, not the measured value.

Qualitative formatting example (format only; never reuse its evidence):
Input target_compound="exampledrug", input_items=["A whole-body PBPK model of exampledrug was developed."]
Output: {"candidates":[{"target_compound":"exampledrug","evidence_type":"public_pbpk_model","parameter":"PBPK model","value":null,"value_low":null,"value_high":null,"value_qualifier":null,"unit":null,"condition":"whole-body PBPK model","source_text":"A whole-body PBPK model of exampledrug was developed.","confidence":0.7,"requires_review":true}]}

Parameter examples:
- Cmax, AUC, Tmax, half_life, clearance, volume_of_distribution, bioavailability
- absorption, distribution, metabolism, excretion, protein_binding
- CYP3A4, CYP2C9, CYP2C19, P-gp, OATP1B1
- renal_impairment, hepatic_impairment, pediatric, geriatric, pregnancy, lactation

Output schema:
{
  "candidates": [
    {
      "target_compound": "midazolam",
      "evidence_type": "clinical_pk",
      "parameter": "Cmax",
      "value": 53,
      "value_low": null,
      "value_high": null,
      "value_qualifier": null,
      "unit": "ng/mL",
      "condition": "10 mg oral dose",
      "source_text": "After a 10 mg oral dose, midazolam reached a mean Cmax of 53 ng/mL.",
      "confidence": 0.72,
      "requires_review": true
    }
  ]
}
"""


# Keep one internally consistent wire contract. The literal-quote legacy prompt
# above remains available for the historical recovery modes. Repeating its
# array-only examples here encouraged silent abstention on supported facts.
LLM_INDEXED_EXTRACTION_PROMPT = """Extract source-backed PBPK evidence about target_compound from input_items.
Use only these source items; do not use outside knowledge. Evaluate each item and
each parameter separately. Return JSON in this order: assessment, reason, candidates.
assessment is extract when at least one explicit target-specific fact is present,
no_evidence for irrelevant text, uncertain for unresolved attribution. Give a short
source-grounded reason. extract requires a nonempty candidates list; the other
decisions require an empty list. Do not discard clear facts because a different
item or parameter is ambiguous. Qualitative facts and numerical intervals are valid.

Each candidate contains the fields in the supplied JSON schema. source_item_index
is the zero-based index of the ONE original item supporting that candidate. The
program restores its exact text. Never combine items, change target_compound, or
assign a comparator, excipient or metabolite's result to the parent drug.

Evidence domains and parameters:
physicochemical: molecular weight, pKa, logP/logD, solubility, permeability, polar
surface area, ionization or formulation properties.
adme: absorption, distribution, metabolism, excretion, bioavailability,
protein_binding, fraction_unbound, blood-to-plasma ratio, intrinsic/renal clearance,
enzyme or transporter substrate/inhibition/induction, stability and efflux.
clinical_pk: Cmax, AUC with its stated window, Tmax, half_life, clearance,
volume_of_distribution, trough, concentration, exposure or accumulation.
ddi: target PK/exposure change or an explicit interaction mechanism involving an
enzyme, transporter or excretion. Name the other drug and the target's role in
condition. A qualitative CYP substrate or inhibition statement needs no numeric PK.
special_population: an explicit PK/ADME result in a population named in the source,
such as renal/hepatic impairment, elderly, pediatric, pregnancy or genotype groups.
public_pbpk_model: source explicitly identifying PBPK/physiologically based/whole-body
modeling, PK-Sim or Simcyp and its public model metadata. Generic PK/PD is insufficient.
Do not emit tissue expression: that domain uses approved structured sources.

Numerical rules:
- A single stated estimate uses value and null value_low/value_high. Do not copy a
  dose, sample size, variability, efficacy/safety endpoint or another parameter.
- A genuine interval for ONE parameter uses value=null and both numeric endpoints.
  Never infer a midpoint. A half-life range or protein-binding range is valid.
- Comparisons written x vs y, versus or respectively are NOT intervals: use a
  qualitative statement with all numeric fields and unit null; do not assign groups.
- Qualitative facts have value/value_low/value_high/value_qualifier/unit all null.
- Copy quantitative values and units exactly. An explicitly mixed time-unit range
  may have both endpoints converted to one of its stated time units. No other
  calculations or missing values. Preserve AUC windows and steady-state distinctions.
- Explicit percent changes and fold changes may be numeric, with direction and
  interacting drug in condition. Bounds use <, <=, >, >= only when stated. Confidence
  intervals preserve their confidence level in value_qualifier, not as measured value.
- Do not call creatinine or dialysis clearance the target's systemic clearance.

The source must assert the result for the target. Reject jointly reported parent
plus metabolite parameters; accept separately attributed parent results. Indications,
efficacy, adverse events, safety warnings, storage, dosing advice, search-hit lists
and failure records alone are not evidence. A DDI requires an explicit PK mechanism
or target exposure change, not a generic warning. Use a parameter named in the
source; mechanism/software/model descriptions belong in condition, never numeric
fields. Set requires_review=true. Emit each supported assertion once. Every output
will independently undergo source, identity, parameter, units and scientific checks.

Field usage examples only (never copy these example facts):
For 'Exampledrug is 55% excreted unchanged in urine', use evidence_type=adme,
parameter=excretion, value=55, unit=%, value_low/value_high/value_qualifier=null.
For 'Otherdrug inhibits CYP3A metabolism of Exampledrug', use evidence_type=ddi,
parameter=CYP3A inhibition, all numeric fields/value_qualifier/unit=null, and put
Otherdrug's inhibition of Exampledrug metabolism in condition. Do not use 'ddi'
as a parameter name. Descriptive prose belongs in condition, never value_qualifier.
"""


EXTRACTION_GENERATION_FORMAT = "pbpk_extraction_schema_v1_xgrammar_0.2.5.post1_ws2"
# Native string bounds stop a decoder from spending its entire generation on
# repeated unit factors or explanations. They constrain only generated labels,
# never the original source text or its indexed transport. Keep these as
# maxLength, not regexes: LMFE retains its bounded free-text token fast path.
EXTRACTION_TEXT_LIMITS = {"parameter": 256, "value_qualifier": 128, "unit": 128, "condition": 2048}


def _bounded_extraction_text(field: str, *, nullable: bool = False) -> dict[str, Any]:
    string = {"type": "string", "maxLength": EXTRACTION_TEXT_LIMITS[field]}
    # LMFE 0.11.3 drops sibling constraints while expanding type:[string,null].
    # Explicit equivalent branches preserve the string bound and fast path.
    return {"anyOf": [string, {"type": "null"}]} if nullable else string


_EXTRACTION_PROPERTIES = {
    "target_compound": {"type": "string"},
    "evidence_type": {"enum": ["physicochemical", "clinical_pk", "adme", "ddi", "special_population", "public_pbpk_model"]},
    "parameter": _bounded_extraction_text("parameter"),
    "value": {"type": ["number", "null"]},
    "value_low": {"type": ["number", "null"]},
    "value_high": {"type": ["number", "null"]},
    "value_qualifier": _bounded_extraction_text("value_qualifier", nullable=True),
    "unit": _bounded_extraction_text("unit", nullable=True),
    "condition": _bounded_extraction_text("condition", nullable=True),
    "source_text": {"type": "string"},
    "confidence": {"type": "number"},
    "requires_review": {"const": True},
}
EXTRACTION_OUTPUT_SCHEMA = {
    "type": "object", "properties": {"candidates": {"type": "array", "items": {
        "type": "object", "properties": _EXTRACTION_PROPERTIES,
        "required": list(_EXTRACTION_PROPERTIES), "additionalProperties": False,
    }}}, "required": ["candidates"], "additionalProperties": False,
}


def extraction_output_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Constrain generation to original quotes; validation still checks provenance."""
    schema = deepcopy(EXTRACTION_OUTPUT_SCHEMA)
    properties = schema["properties"]["candidates"]["items"]["properties"]
    target = payload.get("target_compound")
    if isinstance(target, str) and target:
        properties["target_compound"] = {"const": target}
    items = payload.get("input_items")
    if items is not None:
        if not isinstance(items, list) or not items or any(not isinstance(item, str) or not item for item in items):
            raise ValueError("Extraction input_items must contain complete nonempty source strings")
        properties["source_text"] = {"enum": list(dict.fromkeys(items))}
    elif isinstance(payload.get("input_text_or_table"), str) and payload["input_text_or_table"]:
        properties["source_text"] = {"const": payload["input_text_or_table"]}
    return schema


def indexed_extraction_enabled(payload: dict[str, Any]) -> bool:
    """Keep local generators and explicit historical recovery modes unchanged."""
    return server_backend() and payload.get("generation_recovery_mode") is None


def _extraction_input_items(payload: dict[str, Any]) -> list[str]:
    items = payload.get("input_items")
    if items is None:
        items = [payload.get("input_text_or_table")]
    if not isinstance(items, list) or not items or any(not isinstance(item, str) or not item for item in items):
        raise ValueError("Extraction input_items must contain complete nonempty source strings")
    return items


def extraction_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep indexed model input identical for pilots and provenance-rich runs."""
    if not indexed_extraction_enabled(payload):
        return payload
    items = _extraction_input_items(payload)
    fragment = payload.get("input_text_or_table")
    return {
        "target_compound": payload.get("target_compound"),
        "input_items": list(items),
        "input_text_or_table": fragment if isinstance(fragment, str) else "\n".join(items),
    }


def indexed_extraction_output_schema(payload: dict[str, Any]) -> dict[str, Any]:
    items = _extraction_input_items(payload)
    schema = extraction_output_schema(payload)
    candidate = schema["properties"]["candidates"]["items"]
    candidate["properties"] = {
        ("source_item_index" if key == "source_text" else key):
        ({"type": "integer", "enum": list(range(len(items)))} if key == "source_text" else value)
        for key, value in candidate["properties"].items()
    }
    candidate["required"] = list(candidate["properties"])
    schema["properties"] = {
        "assessment": {"enum": ["extract", "no_evidence", "uncertain"]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 400},
        **schema["properties"],
    }
    schema["required"] = ["assessment", "reason", "candidates"]
    # The model chooses whether to abstain. Once it chooses extract, an empty
    # array is inconsistent, not a successful no-evidence result. Encode this
    # in decoding as well as validation; never force the positive branch.
    positive = deepcopy(schema["properties"]["candidates"])
    positive["minItems"] = 1
    negative = deepcopy(schema["properties"]["candidates"])
    negative["maxItems"] = 0
    schema["oneOf"] = [
        {"type": "object", "properties": {
            "assessment": {"const": "extract"},
            "reason": deepcopy(schema["properties"]["reason"]), "candidates": positive,
        }, "required": [], "additionalProperties": False},
        {"type": "object", "properties": {
            "assessment": {"enum": ["no_evidence", "uncertain"]},
            "reason": deepcopy(schema["properties"]["reason"]), "candidates": negative,
        }, "required": [], "additionalProperties": False},
    ]
    # JSON Schema additionalProperties is local to its subschema.
    for branch in schema["oneOf"]:
        branch["properties"]["reason"] = deepcopy(schema["properties"]["reason"])
        branch["required"] = list(schema["required"])
    return schema


def restore_indexed_extraction(parsed: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Fail the whole response on an invalid reference; never partially accept it."""
    from jsonschema import Draft202012Validator
    if (parsed["assessment"] == "extract") != bool(parsed["candidates"]):
        raise ValueError("Extraction assessment disagrees with the candidate list")
    Draft202012Validator(indexed_extraction_output_schema(payload)).validate(parsed)
    items = _extraction_input_items(payload)
    candidates = []
    for candidate in parsed["candidates"]:
        index = candidate["source_item_index"]
        # JSON Schema treats 1.0 as an integer; the wire protocol requires an
        # actual integer, and Python bool must never be used as a list index.
        if type(index) is not int or not 0 <= index < len(items):
            raise ValueError("source_item_index must be a valid zero-based integer")
        restored = deepcopy(candidate)
        del restored["source_item_index"]
        restored["source_text"] = items[index]
        candidates.append(restored)
    return candidates


REPETITION_RECOVERY_VERSION = "nonrepeating_v1"
REPETITION_RECOVERY_PROMPT = """
Recovery reminder: follow all original evidence rules, especially rules 16 and 34.
Emit each distinct supported assertion only once, then close the candidates array and JSON object.
Do not enumerate a Cartesian product of doses and genotype/group names. Do not duplicate candidate objects.
For multiple paired values followed by respectively or group comparisons, emit qualitative evidence with
value, value_low, value_high, value_qualifier and unit all null, as already required by the original rules.
Preserve the full original input_items quote. Do not add evidence, fill a candidate count, or repeat earlier candidates.
"""


def extraction_prompt(payload: dict[str, Any]) -> str:
    recovery = payload.get("generation_recovery_mode")
    if recovery is None:
        return LLM_INDEXED_EXTRACTION_PROMPT if indexed_extraction_enabled(payload) else LLM_EXTRACTION_PROMPT
    if recovery not in {REPETITION_RECOVERY_VERSION, "comparison_schema_v1", "comparison_sampling_v1", "comparison_penalty_v1", "focused_assertion_v1"}:
        raise ValueError("Unknown evidence generation recovery mode")
    prompt = LLM_EXTRACTION_PROMPT + REPETITION_RECOVERY_PROMPT
    if recovery == "focused_assertion_v1":
        focus = payload["generation_recovery_focus"]
        prompt += (f"\nFor this independent decision, evaluate only input_items[{focus['input_item_index']}] "
                   f"and only its explicitly named parameter {focus['parameter']!r}. Return zero or one candidate "
                   "for this one parameter. Other original items and parameters have separate decisions.\n")
    return prompt


def extraction_recovery_schema(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("generation_recovery_mode") not in {"comparison_schema_v1", "comparison_sampling_v1", "comparison_penalty_v1", "focused_assertion_v1"}:
        return None
    from pbpk_kb.processing.run_llm_extraction import _has_ambiguous_numeric_comparison
    items = payload.get("input_items") or []
    if not items or not any(_has_ambiguous_numeric_comparison(item) for item in items):
        raise ValueError("Comparison recovery requires original comparative assertions")
    focus = payload.get("generation_recovery_focus") if payload.get("generation_recovery_mode") == "focused_assertion_v1" else None
    if focus is not None:
        index, parameter = focus["input_item_index"], focus["parameter"]
        if not isinstance(index, int) or not 0 <= index < len(items) or not parameter or parameter.casefold() not in items[index].casefold():
            raise ValueError("Recovery focus must name a parameter in an original input item")
        items = [items[index]]
    branches = []
    for quote in dict.fromkeys(items):
        ambiguous = _has_ambiguous_numeric_comparison(quote)
        numeric = {"type": "null"} if ambiguous else {"type": ["number", "null"]}
        # Select the exact quote first so its existing numeric rule can constrain
        # the later value fields. Other sentences retain numeric extraction.
        properties = {
            "source_text": {"const": quote},
            "target_compound": {"const": payload["target_compound"]},
            "evidence_type": {"enum": ["physicochemical", "clinical_pk", "adme", "ddi", "special_population", "public_pbpk_model"]},
            "parameter": {"const": focus["parameter"]} if focus is not None else deepcopy(_EXTRACTION_PROPERTIES["parameter"]),
            "value": numeric, "value_low": numeric, "value_high": numeric,
            "value_qualifier": {"type": "null"} if ambiguous else deepcopy(_EXTRACTION_PROPERTIES["value_qualifier"]),
            "unit": {"type": "null"} if ambiguous else deepcopy(_EXTRACTION_PROPERTIES["unit"]),
            "condition": deepcopy(_EXTRACTION_PROPERTIES["condition"]),
            "confidence": {"type": "number"}, "requires_review": {"const": True},
        }
        branches.append({"type": "object", "properties": properties,
                         "required": list(properties), "additionalProperties": False})
    return {"type": "object", "properties": {"candidates": {"type": "array", "items": {"anyOf": branches},
                                                              **({"maxItems": 1} if focus is not None else {})}},
            "required": ["candidates"], "additionalProperties": False}


def has_repeated_complete_candidate(raw_output: str) -> bool:
    """Detect exact repeated objects in a possibly truncated native candidate array."""
    match = re.search(r'"candidates"\s*:\s*\[', raw_output)
    if not match:
        return False
    offset, seen, decoder = match.end(), set(), json.JSONDecoder()
    while True:
        while offset < len(raw_output) and raw_output[offset].isspace():
            offset += 1
        try:
            candidate, offset = decoder.raw_decode(raw_output, offset)
        except (ValueError, json.JSONDecodeError):
            return False
        if not isinstance(candidate, dict):
            return False
        fingerprint = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen:
            return True
        seen.add(fingerprint)
        while offset < len(raw_output) and raw_output[offset].isspace():
            offset += 1
        if offset >= len(raw_output) or raw_output[offset] != ",":
            return False
        offset += 1


@dataclass(frozen=True)
class LLMCallResult:
    status: str
    candidates: list[dict[str, Any]]
    raw_response: str | None
    error: str | None
    model_name: str
    latency_seconds: float | None
    assessment: dict[str, str] | None = None


class LLMExtractionAgent:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 60,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        mode: str = "auto",
    ) -> None:
        validate_backend(base_url, model_name)
        self.api_key = None
        self.model_name, self.base_url = backend_identity()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.mode = mode

    @property
    def prompt_template_version(self) -> str:
        if server_backend():
            config = registered_server()
            backend = ("native-nf4-qwen36" if config.get("backend_kind") == "project_transformers_qwen36_http"
                       else "vllm-qwen")
            return "pbpk-evidence-agent-v20-source-only-assessed-source-index-" + backend + "-" + config["model_config_sha256"][:12]
        return "pbpk-evidence-agent-v14-local-qwen"

    def build_request_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        fragment = str(payload.get("input_text_or_table") or "")
        target_compound = str(payload.get("target_compound") or "").strip().lower()
        indexed = indexed_extraction_enabled(payload)
        source_items = _extraction_input_items(payload) if indexed else None
        model_payload = extraction_model_payload(payload)
        fingerprint = json.dumps(
            {"target_compound": target_compound, "input_text_or_table": fragment},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        recovery_schema = extraction_recovery_schema(payload)
        schema = recovery_schema or (indexed_extraction_output_schema(model_payload) if indexed
                                     else extraction_output_schema(payload))
        return {
            **payload,
            "prompt_template_version": self.prompt_template_version,
            "model_name": self.model_name,
            "generation_format": generation_format("json_schema_xgrammar_0.2.5.post1_ws2" if recovery_schema else EXTRACTION_GENERATION_FORMAT),
            "generation_schema_sha256": sha256(json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest() if schema else None,
            "effective_prompt_sha256": sha256(extraction_prompt(payload).encode("utf-8")).hexdigest(),
            **({"source_item_mapping_sha256": sha256(json.dumps(source_items, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                # Match the actual server user-message serialization, excluding
                # the request record's IDs, paths and generation audit fields.
                "model_payload_sha256": sha256(json.dumps(model_payload, ensure_ascii=False, sort_keys=True,
                                                          separators=(",", ":")).encode("utf-8")).hexdigest(),
                "source_transport": "source_item_index_v1"} if indexed else {}),
            "generation_sampling": ({"do_sample": True, "temperature": .7, "top_p": .8, "top_k": 20,
                                      "min_p": 0.0, "seed": 20260908,
                                      **({"repetition_penalty": 1.1} if payload.get("generation_recovery_mode") == "comparison_penalty_v1" else {})}
                                     if payload.get("generation_recovery_mode") in {"comparison_sampling_v1", "comparison_penalty_v1"}
                                     else {"do_sample": False}),
            "input_hash": sha256(fingerprint.encode("utf-8", errors="replace")).hexdigest(),
        }

    def extract(self, payload: dict[str, Any]) -> LLMCallResult:
        if self.mode == "off":
            return LLMCallResult("skipped_llm_off", [], None, None, self.model_name, None)
        if self.mode not in {"auto", "live"}:
            return LLMCallResult(f"skipped_mode_{self.mode}", [], None, None, self.model_name, None)
        started = time.monotonic()
        raw = None
        try:
            parsed = generate_evidence_json(extraction_prompt(payload), extraction_model_payload(payload))
            raw = json.dumps({"choices": [{"message": {"content": json.dumps(parsed)}}]})
            candidates = (restore_indexed_extraction(parsed, payload) if indexed_extraction_enabled(payload)
                          else parsed.get("candidates"))
            if not isinstance(candidates, list) or any(not isinstance(c, dict) for c in candidates):
                raise ValueError("Local extraction output must contain a list of candidate objects.")
            assessment = ({"decision": parsed["assessment"], "reason": parsed["reason"]}
                          if indexed_extraction_enabled(payload) else None)
            return LLMCallResult("ok", candidates, raw, None, self.model_name, time.monotonic() - started, assessment)
        except Exception as exc:
            return LLMCallResult("failed_local_llm", [], getattr(exc, "raw_response", None) or raw, _sanitize_error(str(exc)), self.model_name, time.monotonic() - started)


def _parse_json_content(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _should_retry_http(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def _sanitize_error(error: str) -> str:
    if error.startswith("CUDA out of memory"):
        return "CUDA out of memory while loading or running project Qwen; request remains unresolved."
    return re.sub(r"sk-[A-Za-z0-9_*\\-]{6,}", "sk-***", error)
