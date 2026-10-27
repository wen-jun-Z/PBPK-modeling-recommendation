from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from pbpk_kb.processing.evidence_llm import (
    DEFAULT_MODEL, LOCAL_BASE_URL, generate_evidence_json, validate_backend,
    backend_identity, generation_format, server_backend, registered_server,
)
from pbpk_kb.processing.auto_quality_stage import build_kg_ready, deterministic_repair, evaluate_evidence
from pbpk_kb.processing.final_evidence_attribution_stage import atomic_write_json, atomic_write_jsonl, iter_jsonl, utc_now


REPAIR_PROMPT = """You repair one PBPK evidence record using only the supplied source context.
Return JSON only with keys: evidence_id, repairs, supporting_text, confidence.
repairs may contain only: value, value_low, value_high, value_qualifier, unit, condition, dose, route, population, reference, victim_drug, perpetrator_drug, enzyme_or_transporter.
Every non-null repaired value must be explicitly supported by the supplied context.
supporting_text must be one exact contiguous quote copied from supplied context.
Choose the shortest sufficient quote (at most 4096 characters); each repaired text field
must be concise (at most 1024 characters). These limits apply to output, never source input.
Do not use outside knowledge. Do not calculate values. Do not guess missing fields.
The presence of a number does not establish its statistic, unit, treatment arm or reference.
Do not label a value mean/median unless the source explicitly identifies that statistic.
Two values compared with vs/versus are separate conditions, never a low/high interval.
Do not assign an ambiguous comparison to one condition. A procedure such as plasmapheresis
is a condition, not a perpetrator_drug. Omit unsupported optional repairs.
If the requested problem cannot be repaired, return repairs={} and an exact quote showing the limitation."""

VALIDATION_PROMPT = """Independently validate one repaired PBPK evidence record against supplied source context.
Ignore any prior explanation and inspect the quote and fields yourself.
Return JSON only with keys:
supported, unsupported_fields, compound_role_correct, parameter_value_unit_correct,
reference_correct, context_sufficient, verbatim_support, confidence, reason.
verbatim_support must be an exact contiguous quote from supplied context.
Choose the shortest sufficient quote (at most 4096 characters); unsupported_fields contains
only field names (at most 128 characters each, at most 32 entries), not explanations.
supported=true only when every repaired scientific field is directly supported, compound attribution is correct,
the parameter/value/unit combination is correct, and any relative effect has an explicit reference.
Number occurrence alone does not support its statistical role, unit, treatment arm or reference.
Reject an inferred mean/median, an invented unit, ambiguous arm assignment, a vs comparison
stored as a range, or a procedure stored as perpetrator_drug. Do not excuse one unsupported
field because other fields are correct. List each unsupported field and set supported=false.
Write reason as one concise sentence of at most 512 characters, without line breaks.
Return the decision directly; do not include deliberation, repeated explanations or JSON advice."""

ALLOWED_REPAIRS = {
    "value", "value_low", "value_high", "value_qualifier", "unit", "condition",
    "dose", "route", "population", "reference", "victim_drug", "perpetrator_drug",
    "enzyme_or_transporter",
}


QUALITY_GENERATION_FORMAT = "pbpk_quality_schema_v2_bounded_reason_source_statistics"


class QualityOutputError(ValueError):
    def __init__(self, message: str, response: Any):
        super().__init__(message)
        self.raw_response = json.dumps(response, ensure_ascii=False)[:65536]


def quality_output_schema(prompt: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    def bounded_text(limit: int, types: Any = "string") -> dict[str, Any]:
        # Native length limits keep LMFE's token fast path and JSON escaping.
        # Long regex bounds disable that path and disallow escaped source quotes.
        return {"type": types, "maxLength": limit}

    if prompt == REPAIR_PROMPT:
        repairs = {key: {"type": ["number", "null"]} if key in {"value", "value_low", "value_high"}
                        else bounded_text(1024, ["string", "number", "null"] if key == "dose" else ["string", "null"])
                   for key in sorted(ALLOWED_REPAIRS)}
        properties = {"evidence_id": {"const": str(payload["evidence_id"])},
            "repairs": {"type": "object", "properties": repairs, "additionalProperties": False},
            "supporting_text": bounded_text(4096), "confidence": {"type": "number", "minimum": 0, "maximum": 1}}
    elif prompt == VALIDATION_PROMPT:
        properties = {"supported": {"type": "boolean"},
            "unsupported_fields": {"type": "array", "items": bounded_text(128), "maxItems": 32},
            **{key: {"type": "boolean"} for key in ("compound_role_correct", "parameter_value_unit_correct", "reference_correct", "context_sufficient")},
            "verbatim_support": bounded_text(4096), "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "minLength": 1, "maxLength": 512}}
    else:
        return None
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def unsupported_numeric_fields(fields: dict[str, Any], context: str) -> list[str]:
    """Require explicit source numbers; substring matches such as 39 vs 390 fail."""
    pattern = r"(?<![\w.])(?:(?<!%)[-+])?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])"
    source_numbers = {Decimal(token) for token in re.findall(pattern, context)}
    converted_interval = False
    quote = fields.get("source_text")
    if isinstance(quote, str) and quote and quote in context:
        from pbpk_kb.processing.run_llm_extraction import LLMEvidenceCandidate, _verified_time_range_conversion
        try:
            candidate = LLMEvidenceCandidate.model_validate({
                "target_compound": "numeric-check", "evidence_type": "clinical_pk",
                "parameter": fields.get("parameter"), "unit": fields.get("unit"),
                "value": fields.get("value"), "value_low": fields.get("value_low"),
                "value_high": fields.get("value_high"), "condition": None,
                "source_text": quote, "confidence": 0, "requires_review": True,
            })
            converted_interval = _verified_time_range_conversion(candidate)
        except (TypeError, ValueError):
            pass
    rejected = []
    if fields.get("value_low") is not None and fields.get("value_high") is not None:
        from types import SimpleNamespace
        from pbpk_kb.processing.run_llm_extraction import _numeric_range_source_error
        interval = SimpleNamespace(**{key: fields.get(key) for key in (
            "value", "value_low", "value_high", "unit", "parameter", "source_text")})
        if not isinstance(quote, str) or not quote or quote not in context or not fields.get("parameter"):
            rejected.extend(["value_low", "value_high"])
        elif _numeric_range_source_error(interval):
            rejected.extend(["value_low", "value_high"])
    for name in ("value", "value_low", "value_high", "dose", "dose_range", "study_count"):
        if converted_interval and name in {"value_low", "value_high"}:
            continue
        value = fields.get(name)
        if value is None or isinstance(value, bool):
            continue
        numbers = [Decimal(token) for token in re.findall(pattern, str(value))]
        if any(number not in source_numbers for number in numbers):
            rejected.append(name)
    return list(dict.fromkeys(rejected))


def unsupported_statistic_fields(fields: dict[str, Any], quote: str) -> list[str]:
    """A reported number alone cannot justify adding a statistical qualifier.

    This is a conservative veto, not proof of the number/statistic relationship;
    the independent field-by-field source validation remains required.
    """
    qualifier = re.sub(r"[_-]+", " ", str(fields.get("value_qualifier") or "").casefold()).strip()
    statistical_terms = {
        "mean": r"\b(?:mean(?!\s+residence\s+time)|average)\b",
        "average": r"\b(?:mean(?!\s+residence\s+time)|average)\b",
        "arithmetic mean": r"\barithmetic\s+mean\b",
        "geometric mean": r"\bgeometric\s+mean\b",
        "median": r"\bmedian\b",
        "sd": r"\b(?:standard\s+deviation|s\.?d\.?)\b",
        "standard deviation": r"\b(?:standard\s+deviation|s\.?d\.?)\b",
        "se": r"\b(?:standard\s+error|s\.?e\.?)\b",
        "standard error": r"\b(?:standard\s+error|s\.?e\.?)\b",
    }
    pattern = statistical_terms.get(qualifier)
    return ["value_qualifier"] if pattern and not re.search(pattern, quote, flags=re.IGNORECASE) else []


def apply_validation_source_guards(
    response: dict[str, Any], fields: dict[str, Any], context: str,
) -> dict[str, Any]:
    response = dict(response)
    numeric_failures = unsupported_numeric_fields(fields, context)
    quote = str(response.get("verbatim_support") or "")
    statistic_failures = unsupported_statistic_fields(fields, quote if quote and quote in context else "")
    response.setdefault("llm_supported_before_numeric_check", response.get("supported"))
    response["deterministic_numeric_failures"] = numeric_failures
    response["deterministic_statistic_failures"] = statistic_failures
    failures = numeric_failures + statistic_failures
    if failures:
        response["supported"] = False
        response["parameter_value_unit_correct"] = False
        response["unsupported_fields"] = sorted(set(response.get("unsupported_fields") or []) | set(failures))
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair and verify PBPK evidence with the registered project Qwen backend.")
    parser.add_argument("--final-run-dir", type=Path, required=True)
    parser.add_argument("--raw-run-dir", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    result = AutoQualityLLMStage(
        final_run_dir=args.final_run_dir,
        raw_run_dir=args.raw_run_dir,
        base_url=args.base_url,
        model=args.model,
        workers=args.workers,
        limit=args.limit,
        reset=args.reset,
    ).run()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


class AutoQualityLLMStage:
    def __init__(
        self,
        *,
        final_run_dir: Path,
        raw_run_dir: Path,
        base_url: str | None = None,
        model: str | None = None,
        workers: int = 16,
        limit: int | None = None,
        reset: bool = False,
    ) -> None:
        self.final_run_dir = final_run_dir
        self.raw_run_dir = raw_run_dir
        self.quality_dir = final_run_dir / "06_auto_quality_control"
        self.output_dir = final_run_dir / "07_llm_quality_validation"
        validate_backend(base_url, model)
        self.model, self.base_url = backend_identity()
        self.backend = (registered_server().get("backend_kind") or "project_registered_vllm") if server_backend() else "project_local_transformers"
        self.workers = max(1, workers)
        self.limit = limit
        self.reset = reset
        self.api_key = None
        self._write_lock = threading.Lock()

    def run(self) -> dict[str, Any]:
        prior_summary = self.output_dir / "quality_summary.json"
        if prior_summary.is_file() and not self.reset:
            prior = json.loads(prior_summary.read_text())
            if prior.get("model") != self.model:
                raise RuntimeError("Use a new run directory for Qwen; do not mix historical evidence-screening models.")
        required = [
            self.quality_dir / "auto_validated_evidence.jsonl",
            self.quality_dir / "repair_pending_evidence.jsonl",
            self.raw_run_dir / "03_chunks_and_embeddings" / "text_chunks.jsonl",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing quality inputs: {missing}")
        if self.output_dir.exists() and self.reset:
            for path in self.output_dir.glob("*.jsonl"):
                path.unlink(missing_ok=True)
            (self.output_dir / "quality_summary.json").unlink(missing_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        already_validated = list(iter_jsonl(self.quality_dir / "auto_validated_evidence.jsonl"))
        initially_quarantined = list(iter_jsonl(self.quality_dir / "quarantined_evidence.jsonl"))
        pending = list(iter_jsonl(self.quality_dir / "repair_pending_evidence.jsonl"))
        context_index = ContextIndex(self.raw_run_dir)
        response_path = self.output_dir / "llm_quality_results.jsonl"
        response_path.touch(exist_ok=True)
        previous = {
            str(row["evidence_id"]): row
            for row in iter_jsonl(response_path)
            if row.get("terminal") is True
        } if response_path.exists() else {}
        if any(row.get("model") != self.model for row in previous.values()):
            raise RuntimeError("Historical quality responses lack matching Qwen provenance; use a new run directory.")
        todo = [row for row in pending if str(row["evidence_id"]) not in previous]
        if self.limit is not None:
            todo = todo[: max(0, self.limit)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._process_one, row, context_index.context_for(row)): row
                for row in todo
            }
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "evidence_id": row["evidence_id"],
                        "terminal": False,
                        "status": "local_llm_error",
                        "error": sanitize(str(exc)),
                        "raw_generation": str(getattr(exc, "raw_response", ""))[:65536],
                    }
                result["model"] = self.model
                with self._write_lock:
                    with response_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

        latest: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(response_path):
            latest[str(row["evidence_id"])] = row
        pending_by_id = {str(row["evidence_id"]): row for row in pending}
        for evidence_id, result in latest.items():
            if result.get("status") != "quarantined_validation_failed":
                continue
            pending_row = pending_by_id.get(evidence_id)
            promoted = promote_from_existing_validation(
                pending_row, result, self.model,
                context=context_index.context_for(pending_row) if pending_row is not None else None,
            )
            if promoted is not None:
                latest[evidence_id] = promoted
        repaired_validated = [row["evidence"] for row in latest.values() if row.get("status") == "auto_validated"]
        for row in repaired_validated:
            row["validation_method"] = "deterministic_plus_independent_llm_v1"
        unresolved = [row for row in latest.values() if row.get("status") != "auto_validated"]
        final_validated = already_validated + repaired_validated
        final_quarantined = initially_quarantined + [
            {
                **next((item for item in pending if str(item["evidence_id"]) == str(result["evidence_id"])), {}),
                "quality_status": "quarantined",
                "quality_reasons": sorted(set(
                    next((item.get("quality_reasons", []) for item in pending if str(item["evidence_id"]) == str(result["evidence_id"])), [])
                    + [str(result.get("status") or "llm_validation_failed")]
                )),
                "llm_quality_result": result,
            }
            for result in unresolved
            if result.get("terminal") is True
        ]
        nodes, edges = build_kg_ready(final_validated)
        atomic_write_jsonl(self.output_dir / "final_quality_decisions.jsonl", latest.values())
        atomic_write_jsonl(self.output_dir / "final_auto_validated_evidence.jsonl", final_validated)
        atomic_write_jsonl(self.output_dir / "final_quarantined_evidence.jsonl", final_quarantined)
        atomic_write_jsonl(self.output_dir / "kg_ready_nodes.jsonl", nodes)
        atomic_write_jsonl(self.output_dir / "kg_ready_edges.jsonl", edges)
        statuses = Counter(str(row.get("status") or "") for row in latest.values())
        summary = {
            "generated_at": utc_now(),
            "model": self.model,
            "base_url": self.base_url,
            "api_key_present": False,
            "backend": self.backend,
            "generation_format": generation_format(QUALITY_GENERATION_FORMAT),
            "independent_verifier_model": False,
            "initial_auto_validated_count": len(already_validated),
            "initial_quarantined_count": len(initially_quarantined),
            "repair_input_count": len(pending),
            "request_limit": self.limit,
            "submitted_count": len(todo),
            "completed_response_count": len(latest),
            "status_counts": dict(sorted(statuses.items())),
            "final_auto_validated_count": len(final_validated),
            "final_quarantined_count": len(final_quarantined),
            "nonterminal_error_count": sum(row.get("terminal") is not True for row in latest.values()),
            "kg_node_count": len(nodes),
            "kg_edge_count": len(edges),
            "accepted": len(latest) == len(pending) and all(row.get("terminal") is True for row in latest.values()),
        }
        summary["partition_complete"] = (
            len(final_validated) + len(final_quarantined)
            == len(already_validated) + len(initially_quarantined) + len(pending)
        )
        summary["accepted"] = summary["accepted"] and summary["partition_complete"]
        atomic_write_json(self.output_dir / "quality_summary.json", summary)
        return summary

    def _process_one(self, row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        evidence_id = str(row["evidence_id"])
        payload = {
            "evidence_id": evidence_id,
            "target_compound": row.get("compound_name"),
            "evidence_type": row.get("evidence_type"),
            "parameter": row.get("parameter"),
            "source_text": row.get("source_text"),
            "current_fields": {
                key: row.get(key)
                for key in ("value", "value_low", "value_high", "value_qualifier", "unit", "condition", "dose", "route", "population")
            },
            "problems": row.get("repairable_blockers") or [],
            "source_context": context,
        }
        repair = self._chat(REPAIR_PROMPT, payload)
        repairs = repair.get("repairs") if isinstance(repair.get("repairs"), dict) else {}
        repairs = {key: value for key, value in repairs.items() if key in ALLOWED_REPAIRS and value not in (None, "")}
        quote = str(repair.get("supporting_text") or "")
        context_text = context["combined_text"]
        if not repairs or not quote or quote not in context_text:
            return {
                "evidence_id": evidence_id,
                "terminal": True,
                "status": "quarantined_unrepairable",
                "repair_response": repair,
            }
        repaired = dict(row)
        metadata = dict(repaired.get("metadata_json") or {})
        for key, value in repairs.items():
            if key in {"reference", "victim_drug", "perpetrator_drug", "enzyme_or_transporter"}:
                metadata[key] = value
            else:
                repaired[key] = value
        repaired["metadata_json"] = metadata
        if {"value", "value_low", "value_high", "unit"} & repairs.keys():
            # These fields are derived, not independent observations. A
            # source-supported correction must not retain the old normalized
            # value or a canonical unit from the previous extraction.
            for key in ("canonical_value", "canonical_value_low", "canonical_value_high", "canonical_unit"):
                repaired.pop(key, None)
        repaired, deterministic_repairs = deterministic_repair(repaired)
        repaired["validation_method"] = "deterministic_plus_independent_llm_v1"
        repaired["llm_repair"] = {
            "model": self.model,
            "repairs": repairs,
            "supporting_text": quote,
            "confidence": repair.get("confidence"),
            "generation_audit": repair.get("generation_audit"),
        }
        validation_payload = {
            "evidence_id": evidence_id,
            "target_compound": repaired.get("compound_name"),
            "original_evidence": payload,
            "repaired_fields": repairs,
            "source_context": context,
        }
        validation = self._chat(VALIDATION_PROMPT, validation_payload)
        validation_quote = str(validation.get("verbatim_support") or "")
        independent_pass = (
            validation.get("supported") is True
            and validation.get("compound_role_correct") is True
            and validation.get("parameter_value_unit_correct") is True
            and validation.get("reference_correct") is True
            and validation.get("context_sufficient") is True
            and validation_quote
            and validation_quote in context_text
            and not validation.get("unsupported_fields")
        )
        decision = evaluate_evidence(repaired, deterministic_repairs)
        if independent_pass and decision["quality_status"] == "auto_validated":
            repaired.update(decision)
            repaired["quality_status"] = "auto_validated"
            repaired["llm_validation_status"] = "passed"
            repaired["llm_validation"] = {"model": self.model, **validation}
            return {
                "evidence_id": evidence_id,
                "terminal": True,
                "status": "auto_validated",
                "evidence": repaired,
            }
        return {
            "evidence_id": evidence_id,
            "terminal": True,
            "status": "quarantined_validation_failed",
            "repair_response": repair,
            "validation_response": validation,
            "rule_decision": decision,
        }

    def _chat(self, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = generate_evidence_json(system_prompt, payload)
        schema = quality_output_schema(system_prompt, payload)
        if schema is not None:
            from jsonschema import Draft202012Validator
            from hashlib import sha256
            errors = list(Draft202012Validator(schema).iter_errors(response))
            if errors:
                locations = ["/".join(map(str, error.path)) or "root" for error in errors]
                raise QualityOutputError("Invalid quality response fields: " + ", ".join(locations), response)
            if system_prompt == VALIDATION_PROMPT and any(c in response["reason"] for c in "\r\n"):
                raise QualityOutputError("Quality reason must be a single line", response)
            digest = lambda value: sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            response["generation_audit"] = {"generation_format": generation_format(QUALITY_GENERATION_FORMAT),
                "request_sha256": digest(payload), "schema_sha256": digest(schema),
                "prompt_sha256": sha256(system_prompt.encode()).hexdigest()}
        if system_prompt == VALIDATION_PROMPT:
            context = (payload.get("source_context") or {}).get("combined_text", "")
            original = payload.get("original_evidence") or {}
            fields = {**(original.get("current_fields") or {}), **(payload.get("repaired_fields") or {}),
                      "parameter": original.get("parameter"), "source_text": original.get("source_text")}
            response = apply_validation_source_guards(response, fields, context)
        return response


class ContextIndex:
    def __init__(self, raw_run_dir: Path) -> None:
        self.by_id: dict[str, dict[str, Any]] = {}
        self.by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in iter_jsonl(raw_run_dir / "03_chunks_and_embeddings" / "text_chunks.jsonl"):
            self.by_id[str(row["chunk_id"])] = row
            self.by_document[str(row.get("parsed_document_id") or "")].append(row)
        for rows in self.by_document.values():
            rows.sort(key=lambda item: int(item.get("chunk_index") or 0))

    def context_for(self, evidence: dict[str, Any]) -> dict[str, Any]:
        current = self.by_id.get(str(evidence.get("chunk_id") or ""))
        selected: list[dict[str, Any]] = []
        if current:
            rows = self.by_document.get(str(current.get("parsed_document_id") or ""), [])
            position = next((index for index, row in enumerate(rows) if row["chunk_id"] == current["chunk_id"]), -1)
            if position >= 0:
                selected = rows[max(0, position - 1) : position + 2]
        if not selected:
            selected = [{
                "chunk_id": evidence.get("chunk_id"),
                "chunk_index": None,
                "chunk_text": evidence.get("source_text"),
                "source_location": evidence.get("source_location"),
            }]
        chunks = [
            {
                "chunk_id": row.get("chunk_id"),
                "chunk_index": row.get("chunk_index"),
                "source_location": row.get("source_location"),
                "text": row.get("chunk_text"),
            }
            for row in selected
        ]
        return {
            "chunks": chunks,
            "combined_text": "\n\n".join(str(row.get("text") or "") for row in chunks),
        }


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM response is not a JSON object")
    return value


def promote_from_existing_validation(
    row: dict[str, Any] | None,
    result: dict[str, Any],
    model: str = DEFAULT_MODEL,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if row is None or context is None:
        return None
    repair = result.get("repair_response") if isinstance(result.get("repair_response"), dict) else {}
    validation = result.get("validation_response") if isinstance(result.get("validation_response"), dict) else {}
    repairs = repair.get("repairs") if isinstance(repair.get("repairs"), dict) else {}
    repairs = {key: value for key, value in repairs.items() if key in ALLOWED_REPAIRS and value not in (None, "")}
    context_text = str(context.get("combined_text") or "")
    repair_quote = str(repair.get("supporting_text") or "")
    validation_quote = str(validation.get("verbatim_support") or "")
    if not repair_quote or repair_quote not in context_text or not validation_quote or validation_quote not in context_text:
        return None
    validation = apply_validation_source_guards(validation, {**row, **repairs}, context_text)
    independent_pass = (
        validation.get("supported") is True
        and validation.get("compound_role_correct") is True
        and validation.get("parameter_value_unit_correct") is True
        and validation.get("reference_correct") is True
        and validation.get("context_sufficient") is True
        and bool(validation.get("verbatim_support"))
        and not validation.get("unsupported_fields")
    )
    if not repairs or not independent_pass:
        return None
    repaired = dict(row)
    metadata = dict(repaired.get("metadata_json") or {})
    for key, value in repairs.items():
        if key in {"reference", "victim_drug", "perpetrator_drug", "enzyme_or_transporter"}:
            metadata[key] = value
        else:
            repaired[key] = value
    repaired["metadata_json"] = metadata
    if {"value", "value_low", "value_high", "unit"} & repairs.keys():
        for key in ("canonical_value", "canonical_value_low", "canonical_value_high", "canonical_unit"):
            repaired.pop(key, None)
    repaired, deterministic_repairs = deterministic_repair(repaired)
    decision = evaluate_evidence(repaired, deterministic_repairs)
    if decision["quality_status"] != "auto_validated":
        return None
    repaired.update(decision)
    repaired.update({
        "quality_status": "auto_validated",
        "validation_method": "deterministic_plus_independent_llm_v1",
        "llm_validation_status": "passed",
        "llm_repair": {
            "model": model,
            "repairs": repairs,
            "supporting_text": repair.get("supporting_text"),
            "confidence": repair.get("confidence"),
            "generation_audit": repair.get("generation_audit"),
        },
        "llm_validation": {"model": model, **validation},
    })
    return {
        "evidence_id": row["evidence_id"],
        "terminal": True,
        "status": "auto_validated",
        "evidence": repaired,
        "promoted_by_rule_recheck": True,
    }


def sanitize(value: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]{6,}", "sk-***", value)


if __name__ == "__main__":
    main()
