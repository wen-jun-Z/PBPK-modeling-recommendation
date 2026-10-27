from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.graph.final_evidence_graph import extract_graph_facts
from pbpk_kb.normalization.unit_normalizer import normalize_unit, to_canonical_value
from pbpk_kb.processing.final_evidence_attribution_stage import atomic_write_json, atomic_write_jsonl, iter_jsonl, utc_now
from pbpk_kb.processing.structured_evidence_validation import validate_structured_evidence
from pbpk_kb.processing.table_evidence_validation import validate_table_evidence


ALLOWED_TYPES = {
    "physicochemical",
    "adme",
    "clinical_pk",
    "ddi",
    "special_population",
    "tissue_expression",
    "public_pbpk_model",
}
FINAL_ROLES = {"target_compound", "victim_drug", "perpetrator_drug"}
QUALITATIVE_TYPES = {"adme", "public_pbpk_model"}
RELATIVE_RE = re.compile(
    r"\b(?:unchanged|increase[ds]?|decrease[ds]?|reduc(?:e[ds]?|tion)|change[ds]?|difference|"
    r"ratio|fold|versus|vs\.?|compared with|relative to)\b",
    re.IGNORECASE,
)
RESULT_INTENT_RE = re.compile(
    r"\b(?:purpose|objective|aim)\s+(?:of\s+[^.]{0,60}\s+)?(?:is|was|were|to)\b|"
    r"\b(?:study|trial)\s+(?:is|was)\s+(?:designed|conducted)\s+to\b|"
    r"\bto\s+(?:assess|evaluate|determine|investigate|examine)\b",
    re.IGNORECASE,
)
REFERENCE_RE = re.compile(
    r"\b(?:placebo|control|baseline|reference|healthy|normal|without|alone|versus|vs\.?|"
    r"compared with|relative to|coadministered with|in combination with)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:\s*(?:to|[-–])\s*[-+]?\d+(?:\.\d+)?)?")
UNIT_RE = re.compile(
    r"(?P<unit>ng\s*[·*x]?\s*h(?:r)?\s*/\s*mL|(?:u|µ|μ|mc|m)g\s*[·*x]?\s*h(?:r)?\s*/\s*(?:mL|L)|"
    r"(?:ng|ug|µg|μg|mcg|mg)\s*/\s*(?:mL|L)|mL\s*/\s*min(?:\s*/\s*kg)?|"
    r"L\s*/\s*h(?:r)?(?:\s*/\s*kg)?|(?:min|minutes?|h|hr|hours?|days?)|"
    r"(?:mg|ug|µg|μg|mcg)\s*/\s*kg|g\s*/\s*mol|nM|uM|µM|μM|mM|%|fold|ratio)",
    re.IGNORECASE,
)
PARAMETER_DIMENSIONS = {
    "auc": {"auc"},
    "cmax": {"concentration"},
    "concentration": {"concentration", "molar_concentration"},
    "half_life": {"time"},
    "halflife": {"time"},
    "tmax": {"time"},
    "clearance": {"clearance", "clearance_per_weight", "clearance_bsa_normalized"},
    "volume_of_distribution": {"volume", "volume_per_weight"},
    "molecularweight": {"molar_mass"},
    "molecular_weight": {"molar_mass"},
    "bioavailability": {"fraction_percent", "dimensionless_ratio"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build automatically validated PBPK evidence and KG-ready facts.")
    parser.add_argument("--final-run-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    result = AutoQualityStage(
        final_run_dir=args.final_run_dir,
        input_path=args.input,
        output_dir=args.output_dir,
        reset=args.reset,
    ).run()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


class AutoQualityStage:
    def __init__(
        self,
        *,
        final_run_dir: Path,
        input_path: Path | None = None,
        output_dir: Path | None = None,
        reset: bool = False,
    ) -> None:
        self.final_run_dir = final_run_dir
        self.input_path = input_path or final_run_dir / "05_evidence_layers" / "reviewable_evidence.jsonl"
        self.output_dir = output_dir or final_run_dir / "06_auto_quality_control"
        self.reset = reset

    def run(self) -> dict[str, Any]:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Evidence input does not exist: {self.input_path}")
        summary_path = self.output_dir / "quality_summary.json"
        if summary_path.exists() and not self.reset:
            raise RuntimeError("Auto quality output already exists; use --reset for an intentional rebuild.")

        rows = list(iter_jsonl(self.input_path))
        decisions: list[dict[str, Any]] = []
        validated: list[dict[str, Any]] = []
        repair_pending: list[dict[str, Any]] = []
        quarantined: list[dict[str, Any]] = []
        repair_requests: list[dict[str, Any]] = []

        for row in rows:
            repaired, repairs = deterministic_repair(row)
            decision = evaluate_evidence(repaired, repairs)
            decisions.append(
                {
                    "evidence_id": repaired.get("evidence_id"),
                    "compound_id": repaired.get("compound_id"),
                    "source_id": repaired.get("source_id"),
                    "source_location": repaired.get("source_location"),
                    **decision,
                }
            )
            output = {**repaired, **decision, "deterministic_repairs": repairs}
            status = decision["quality_status"]
            if status == "auto_validated":
                validated.append(output)
            elif status == "repair_pending":
                repair_pending.append(output)
                repair_requests.append(build_repair_request(output))
            else:
                quarantined.append(output)

        nodes, edges = build_kg_ready(validated)
        counts = Counter(row["quality_status"] for row in decisions)
        reason_counts = Counter(reason for row in decisions for reason in row["quality_reasons"])
        summary = {
            "generated_at": utc_now(),
            "input_path": str(self.input_path),
            "input_count": len(rows),
            "status_counts": dict(sorted(counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "repair_request_count": len(repair_requests),
            "kg_node_count": len(nodes),
            "kg_edge_count": len(edges),
            "accepted": len(rows) == len(decisions) == sum(counts.values()),
            "outputs": {
                "decisions": "quality_decisions.jsonl",
                "auto_validated": "auto_validated_evidence.jsonl",
                "repair_pending": "repair_pending_evidence.jsonl",
                "quarantined": "quarantined_evidence.jsonl",
                "repair_requests": "context_repair_requests.jsonl",
                "kg_nodes": "kg_ready_nodes.jsonl",
                "kg_edges": "kg_ready_edges.jsonl",
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(self.output_dir / "quality_decisions.jsonl", decisions)
        atomic_write_jsonl(self.output_dir / "auto_validated_evidence.jsonl", validated)
        atomic_write_jsonl(self.output_dir / "repair_pending_evidence.jsonl", repair_pending)
        atomic_write_jsonl(self.output_dir / "quarantined_evidence.jsonl", quarantined)
        atomic_write_jsonl(self.output_dir / "context_repair_requests.jsonl", repair_requests)
        atomic_write_jsonl(self.output_dir / "kg_ready_nodes.jsonl", nodes)
        atomic_write_jsonl(self.output_dir / "kg_ready_edges.jsonl", edges)
        atomic_write_json(summary_path, summary)
        return summary


def deterministic_repair(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repaired = dict(row)
    repairs: list[dict[str, Any]] = []
    table = validate_table_evidence(repaired)
    if table is not None and table["accepted"] and repaired.get("value_qualifier") is None:
        qualifier = table["audit"].get("value_qualifier")
        if qualifier is not None:
            repaired["value_qualifier"] = qualifier
            repairs.append(repair_record("value_qualifier", None, qualifier, str(repaired.get("source_text") or ""),
                                         repaired, "verified_source_table_cell"))
    structured = validate_structured_evidence(repaired)
    if structured is not None:
        metadata = repaired.get("metadata_json")
        repaired["metadata_json"] = {**(metadata if isinstance(metadata, dict) else {}),
                                     "structured_source_validation": structured["audit"]}
        if not structured["accepted"]:
            # Do not convert a value under a unit already contradicted by the
            # source schema, or otherwise transform an unverified source fact.
            return repaired, repairs
        for field, new in (("unit", structured["unit"]), ("canonical_unit", structured["unit"]),
                           ("unit_dimension", structured["dimension"])):
            old = repaired.get(field)
            if old != new:
                repaired[field] = new
                repairs.append(repair_record(field, old, new, str(repaired.get("source_text") or ""),
                                             repaired, "verified_provider_property_schema"))
    value = numeric_value(repaired)
    unit = clean(repaired.get("unit"))
    source_text = clean(repaired.get("source_text"))
    interval = numeric_range(repaired)
    if (value is not None or interval is not None) and not unit:
        matches = {normalize_space(match.group("unit")) for match in UNIT_RE.finditer(source_text)}
        if len(matches) == 1:
            unit = next(iter(matches))
            repaired["unit"] = unit
            repairs.append(repair_record("unit", None, unit, source_text, repaired, "unique_unit_in_source_text"))
    if value is not None and unit:
        canonical_value, canonical_unit = to_canonical_value(value, unit)
        if repaired.get("canonical_value") is None and canonical_value is not None:
            repaired["canonical_value"] = canonical_value
            repairs.append(repair_record("canonical_value", None, canonical_value, source_text, repaired, "unit_normalizer"))
        if not repaired.get("canonical_unit") and canonical_unit:
            repaired["canonical_unit"] = canonical_unit
            repairs.append(repair_record("canonical_unit", None, canonical_unit, source_text, repaired, "unit_normalizer"))
    if interval is not None and unit:
        for field, endpoint in zip(("value_low", "value_high"), interval):
            canonical, canonical_unit = to_canonical_value(endpoint, unit)
            key = "canonical_" + field
            if repaired.get(key) is None:
                repaired[key] = canonical
                repairs.append(repair_record(key, None, canonical, source_text, repaired, "unit_normalizer"))
        if not repaired.get("canonical_unit"):
            repaired["canonical_unit"] = canonical_unit
    return repaired, repairs


def evaluate_evidence(row: dict[str, Any], repairs: list[dict[str, Any]]) -> dict[str, Any]:
    fatal: list[str] = []
    repairable: list[str] = []
    warnings: list[str] = []
    missing_context: list[str] = []
    source_text_raw = str(row.get("source_text") or "")
    source_text = source_text_raw.strip()

    required = (
        "evidence_id", "compound_id", "source_id", "raw_asset_id", "storage_path",
        "checksum", "source_location", "source_text", "source_text_hash",
        "extraction_method", "extractor_version",
    )
    missing = [field for field in required if row.get(field) in (None, "")]
    if missing:
        fatal.append("missing_provenance:" + ",".join(missing))
    if source_text:
        digest = hashlib.sha256(source_text_raw.encode("utf-8", errors="replace")).hexdigest()
        if digest != row.get("source_text_hash"):
            fatal.append("source_text_hash_mismatch")
    storage_path = clean(row.get("storage_path"))
    if storage_path and not Path(storage_path).exists():
        fatal.append("storage_path_not_found")

    evidence_type = clean(row.get("evidence_type")).lower()
    if evidence_type not in ALLOWED_TYPES:
        fatal.append("unsupported_evidence_type")
    role = clean(row.get("compound_role"))
    if role not in FINAL_ROLES:
        fatal.append("non_final_compound_role")
    if not clean(row.get("parameter")):
        fatal.append("missing_parameter")

    value = numeric_value(row)
    interval = numeric_range(row)
    has_interval_fields = row.get("value_low") is not None or row.get("value_high") is not None
    if has_interval_fields and interval is None:
        fatal.append("invalid_numeric_range")
    if has_interval_fields and value is not None:
        fatal.append("mixed_scalar_and_range")
    has_numeric = value is not None or interval is not None
    structured = validate_structured_evidence(row)
    table = validate_table_evidence(row)
    if structured is not None and not structured["accepted"]:
        fatal.append(structured["error"])
    if table is not None and not table["accepted"]:
        fatal.append(table["error"])
    if has_numeric and structured is None and table is None:
        from pbpk_kb.processing.run_llm_extraction import _numeric_parameter_owner_error
        target = clean(row.get("attributed_compound_name") or row.get("compound_name") or row.get("target_compound") or row.get("compound"))
        numeric_owner_error = _numeric_parameter_owner_error(row, target)
        if numeric_owner_error:
            fatal.append(numeric_owner_error)
    unit = clean(row.get("canonical_unit") or row.get("unit"))
    qualitative = evidence_type in QUALITATIVE_TYPES and not has_numeric
    text_looks_quantitative = quantitative_fact_in_text(row, source_text)
    if has_numeric and not unit:
        repairable.append("numeric_value_missing_unit")
    if not has_numeric and text_looks_quantitative and not qualitative:
        repairable.append("quantitative_statement_missing_value")
    if has_numeric and unit and not unit_dimension_matches(row, unit):
        fatal.append("parameter_unit_dimension_mismatch")

    for field in ("dose", "route", "population", "condition"):
        if row.get(field) in (None, ""):
            missing_context.append(field)
    # "Excreted unchanged" describes parent-drug recovery, not a change
    # relative to a missing comparator. Other uses of unchanged remain gated.
    relative_text = re.sub(r"\b(?:excreted|eliminated)\s+unchanged\b", "parent drug excretion", source_text, flags=re.I)
    relative = bool(RELATIVE_RE.search(relative_text)) or unit.lower() in {"ratio", "fold"}
    metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
    reference_supported = bool(
        metadata.get("reference")
        or REFERENCE_RE.search(" ".join((source_text, clean(row.get("condition")))))
    )
    if relative and not reference_supported:
        repairable.append("relative_effect_missing_reference")

    if evidence_type == "ddi":
        has_mechanism = bool(re.search(r"\b(?:CYP\d|UGT\d|P-?gp|BCRP|OATP|OCT|MATE|inhibit|induc|substrate)\b", source_text, re.I))
        has_interacting_drug = ddi_has_counterparty(row, source_text)
        has_effect = bool(
            RELATIVE_RE.search(source_text)
            or re.search(r"\b(?:AUC|Cmax|exposure|clearance|bioavailability|pharmacokinetics?|disposition)\b", source_text, re.I)
        )
        if RESULT_INTENT_RE.search(source_text) and not re.search(
            r"\b(?:resulted|observed|found|showed|increased|decreased|reduced|unchanged|no significant effect)\b",
            source_text,
            re.I,
        ):
            fatal.append("study_intent_without_observed_result")
        if not has_interacting_drug and not has_mechanism:
            repairable.append("ddi_missing_counterparty_or_mechanism")
        if not has_effect:
            fatal.append("ddi_missing_pk_effect_or_mechanism")

    conflict_statuses = {clean(item) for item in row.get("conflict_statuses") or []}
    if conflict_statuses & {"unresolved_conflict", "insufficient_context"}:
        fatal.append("unresolved_conflict")

    reasons = sorted(set(fatal + repairable + warnings))
    if fatal:
        status = "quarantined"
    elif repairable:
        status = "repair_pending"
    else:
        status = "auto_validated"
    return {
        "quality_status": status,
        "quality_reasons": reasons,
        "fatal_blockers": sorted(set(fatal)),
        "repairable_blockers": sorted(set(repairable)),
        "quality_warnings": sorted(set(warnings)),
        "context_missing_fields": missing_context,
        "reference_supported": reference_supported if relative else None,
        "validation_method": "deterministic_quality_gate_v1",
        "llm_validation_required": bool(repairable),
        "llm_validation_status": "not_required" if not repairable else "pending",
        "quality_checked_at": utc_now(),
        "repair_count": len(repairs),
        **({"structured_source_validation": structured["audit"]} if structured is not None else {}),
        **({"table_source_validation": table["audit"]} if table is not None else {}),
    }


def build_repair_request(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": row["evidence_id"],
        "compound_id": row.get("compound_id"),
        "source_id": row.get("source_id"),
        "source_location": row.get("source_location"),
        "storage_path": row.get("storage_path"),
        "chunk_id": row.get("chunk_id"),
        "parsed_document_id": row.get("parsed_document_id"),
        "parsed_table_id": row.get("parsed_table_id"),
        "requested_fields": row.get("repairable_blockers") or [],
        "context_scope": ["current_source_text", "table_header_and_footnote", "previous_chunk", "next_chunk"],
        "instruction": "Return only verbatim-supported fields with supporting_text and supporting_location; do not infer.",
        "status": "pending",
    }


def build_kg_ready(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[tuple[str, str], dict[str, Any]] = {}
    edges: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        compound_id = clean(row.get("compound_id"))
        evidence_id = clean(row.get("evidence_id"))
        parameter = clean(row.get("parameter"))
        add_node(nodes, "Compound", compound_id, {"name": row.get("compound_name")})
        add_node(nodes, "Evidence", evidence_id, {"evidence_type": row.get("evidence_type"), "quality_status": "auto_validated"})
        add_node(nodes, "Parameter", parameter.lower(), {"name": parameter})
        provenance = {
            "evidence_id": evidence_id,
            "source_id": row.get("source_id"),
            "source_location": row.get("source_location"),
            "source_text_hash": row.get("source_text_hash"),
            "quality_status": "auto_validated",
            "validation_method": row.get("validation_method"),
            "confidence": row.get("extraction_confidence"),
        }
        add_edge(edges, "Compound", compound_id, "HAS_EVIDENCE", "Evidence", evidence_id, provenance)
        add_edge(edges, "Evidence", evidence_id, "HAS_PARAMETER", "Parameter", parameter.lower(), provenance)
        source_id = clean(row.get("source_id"))
        add_node(nodes, "SourceDocument", source_id, {"source_id": source_id})
        add_edge(edges, "Evidence", evidence_id, "SUPPORTED_BY", "SourceDocument", source_id, provenance)
        population = clean(row.get("population"))
        if population:
            population_id = stable_graph_id("population", population)
            add_node(nodes, "Population", population_id, {"name": population})
            add_edge(edges, "Evidence", evidence_id, "STUDIED_IN", "Population", population_id, provenance)
        context_properties = {
            key: row.get(key)
            for key in ("condition", "dose", "route")
            if row.get(key) not in (None, "")
        }
        if context_properties:
            context_id = stable_graph_id("study_context", json.dumps(context_properties, ensure_ascii=False, sort_keys=True))
            add_node(nodes, "StudyContext", context_id, context_properties)
            add_edge(edges, "Evidence", evidence_id, "HAS_STUDY_CONTEXT", "StudyContext", context_id, provenance)
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        if row.get("evidence_type") == "public_pbpk_model":
            model_name = clean(
                metadata.get("model_name")
                or recursive_value(metadata, ("model_name", "model", "repository_path"))
                or row.get("parameter")
            )
            if model_name:
                model_id = stable_graph_id("public_model", model_name)
                add_node(
                    nodes,
                    "PublicModel",
                    model_id,
                    {
                        "name": model_name,
                        "software": recursive_value(metadata, ("software", "platform", "tool")),
                    },
                )
                add_edge(edges, "Compound", compound_id, "DESCRIBED_BY_MODEL", "PublicModel", model_id, provenance)
        if row.get("evidence_type") == "ddi":
            counterpart = clean(
                metadata.get("perpetrator_drug")
                or metadata.get("victim_drug")
                or metadata.get("interacting_drug")
            )
            if counterpart and counterpart.lower() != clean(row.get("compound_name")).lower():
                counterpart_id = stable_graph_id("compound_name", counterpart)
                add_node(nodes, "Compound", counterpart_id, {"name": counterpart, "identity_status": "name_only"})
                add_edge(edges, "Compound", compound_id, "HAS_DDI_WITH", "Compound", counterpart_id, provenance)
        for fact in extract_graph_facts(row):
            node_type = "Enzyme" if fact.target.target_type == "enzyme" else "Transporter"
            add_node(nodes, node_type, fact.target.key, {"name": fact.target.preferred_name, "gene_symbol": fact.target.gene_symbol})
            add_edge(
                edges, "Compound", compound_id, fact.relationship_type.upper(), node_type, fact.target.key,
                {**provenance, "confidence": fact.confidence, "supporting_sentence": fact.evidence_sentence},
            )
    return list(nodes.values()), list(edges.values())


def recursive_value(value: Any, keys: tuple[str, ...]) -> Any:
    wanted = {key.lower() for key in keys}
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in wanted and item not in (None, "", [], {}):
                return item
        for item in value.values():
            found = recursive_value(item, keys)
            if found not in (None, ""):
                return found
    elif isinstance(value, list):
        for item in value:
            found = recursive_value(item, keys)
            if found not in (None, ""):
                return found
    return None


def stable_graph_id(kind: str, value: str) -> str:
    return hashlib.sha256(f"{kind}:{value.strip().lower()}".encode("utf-8")).hexdigest()[:24]


def add_node(store: dict[tuple[str, str], dict[str, Any]], node_type: str, node_id: str, properties: dict[str, Any]) -> None:
    if node_id:
        store.setdefault((node_type, node_id), {"node_type": node_type, "node_id": node_id, "properties": properties})


def add_edge(
    store: dict[tuple[str, str, str, str], dict[str, Any]],
    from_type: str,
    from_id: str,
    relationship: str,
    to_type: str,
    to_id: str,
    properties: dict[str, Any],
) -> None:
    if from_id and to_id:
        key = (from_id, relationship, to_id, clean(properties.get("evidence_id")))
        store.setdefault(
            key,
            {
                "from_type": from_type,
                "from_id": from_id,
                "relationship_type": relationship,
                "to_type": to_type,
                "to_id": to_id,
                "properties": properties,
            },
        )


def unit_dimension_matches(row: dict[str, Any], unit: str) -> bool:
    parameter = re.sub(r"[^a-z0-9]+", "_", clean(row.get("parameter")).lower()).strip("_")
    dimension = normalize_unit(unit).dimension
    source_text = clean(row.get("source_text"))
    if dimension in {"fraction_percent", "relative_change_percent", "dimensionless_ratio"} and (
        any(term in parameter for term in ("change", "ratio", "fold", "difference", "increase", "decrease"))
        or bool(RELATIVE_RE.search(source_text))
    ):
        return True
    expected = next((dimensions for key, dimensions in PARAMETER_DIMENSIONS.items() if key == parameter or key in parameter), None)
    if expected is None:
        return True
    return dimension in expected


def parameter_is_quantitative(row: dict[str, Any]) -> bool:
    parameter = clean(row.get("parameter")).lower()
    return any(term in parameter for term in ("auc", "cmax", "tmax", "half", "clearance", "volume", "weight", "logp", "pka", "solubility", "ratio"))


def quantitative_fact_in_text(row: dict[str, Any], source_text: str) -> bool:
    if not parameter_is_quantitative(row):
        return False
    parameter = clean(row.get("parameter"))
    parameter_tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", parameter) if len(token) >= 3]
    if parameter_tokens and not any(re.search(rf"\b{re.escape(token)}\b", source_text, re.I) for token in parameter_tokens):
        return False
    for number in NUMBER_RE.finditer(source_text):
        following = source_text[number.end() : number.end() + 30]
        if UNIT_RE.match(following.lstrip()):
            return True
    return False


def ddi_has_counterparty(row: dict[str, Any], source_text: str) -> bool:
    metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
    if any(metadata.get(key) for key in ("victim_drug", "perpetrator_drug", "interacting_drug")):
        return True
    condition = clean(row.get("condition"))
    if re.search(r"\b(?:with|versus|vs\.?|coadministration with|in combination with)\s+[A-Za-z]", condition, re.I):
        return True
    target = clean(row.get("compound_name") or row.get("attributed_compound_name"))
    drug_like = re.findall(r"\b[A-Za-z][A-Za-z-]{4,}(?:mab|nib|azole|mycin|vir|statin|pril|sartan|olol|caine|zepam|formin)\b", source_text, re.I)
    return any(item.lower() != target.lower() for item in drug_like)


def numeric_range(row: dict[str, Any]) -> tuple[float, float] | None:
    try:
        if any(isinstance(row.get(key), bool) for key in ("value_low", "value_high")):
            return None
        low, high = float(row["value_low"]), float(row["value_high"])
        return (low, high) if math.isfinite(low) and math.isfinite(high) and low <= high else None
    except (KeyError, TypeError, ValueError):
        return None


def numeric_value(row: dict[str, Any]) -> float | None:
    for key in ("canonical_value", "value"):
        try:
            if row.get(key) is not None:
                return float(row[key])
        except (TypeError, ValueError):
            pass
    return None


def repair_record(field: str, old: Any, new: Any, text: str, row: dict[str, Any], method: str) -> dict[str, Any]:
    return {
        "field": field,
        "old_value": old,
        "new_value": new,
        "supporting_text": text,
        "supporting_location": row.get("source_location"),
        "repair_method": method,
    }


def clean(value: Any) -> str:
    return str(value or "").strip()


def normalize_space(value: str) -> str:
    return " ".join(value.split())


if __name__ == "__main__":
    main()
