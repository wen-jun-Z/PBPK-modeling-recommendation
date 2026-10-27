from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pbpk_kb.recommendation.entity_extraction import normalize_text
from pbpk_kb.recommendation.local_llm import (
    DEFAULT_LOCAL_LLM,
    LocalStructuredGenerator,
    StructuredGeneration,
)


PROMPT_VERSION = "pbpk-complex-ddi-entity-extractor-1.1.0"
VALIDATOR_VERSION = "pbpk-complex-ddi-entity-validator-1.1.0"
MIN_CONFIDENCE = 0.80
ENTITY_TYPES = {
    "specific_compound",
    "drug_class",
    "botanical_product",
    "food",
    "combination_product",
    "unknown_abbreviation",
}
SYSTEM_PROMPT = """Extract non-target interacting products from one PBPK drug-interaction passage.
Return exactly one JSON object with an `entities` array. Each entity must contain mention_text,
entity_type, role, supporting_quote, confidence, and reason. mention_text and supporting_quote
must be exact contiguous substrings of source_text. entity_type must be one of:
specific_compound, drug_class, botanical_product, food, combination_product,
unknown_abbreviation. Use specific_compound only for one named chemical or drug, not for classes,
foods, herbs, extracts, regimens, abbreviations whose expansion is absent, or the target compound.
role must be victim_drug, perpetrator_drug, interacting_product, or unknown. confidence must be a
JSON number from 0 to 1. An all-uppercase token of five or fewer letters without an expansion in
source_text must be unknown_abbreviation. For a supporting quote, copy the entire source_text when
a shorter exact quote is uncertain. Do not create identifiers, expand abbreviations, or use outside
knowledge."""

DRUG_CLASS_TERMS = re.compile(
    r"\b(?:quinolones?|fluoroquinolones?|antibiotics?|antimycotics?|"
    r"blockers?|inhibitors?|inducers?|substrates?)\b",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract unresolved DDI counterpart mentions with a local LLM.")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_LOCAL_LLM)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run_extraction(
        tasks_path=args.tasks,
        output_dir=args.output_dir,
        model_name=args.model,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def run_extraction(
    *,
    tasks_path: Path,
    output_dir: Path,
    model_name: str,
    resume: bool,
) -> dict[str, Any]:
    tasks = [
        row for row in _iter_jsonl(tasks_path)
        if row.get("routing") == "unresolved_no_counterpart"
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "complex_entity_state.jsonl"
    completed = {
        str(row["evidence_id"]): row for row in _iter_jsonl(state_path)
    } if resume and state_path.exists() else {}
    if state_path.exists() and not resume:
        raise RuntimeError(f"State exists at {state_path}; use --resume or a new output directory.")
    generator = LocalStructuredGenerator(model_name) if len(completed) < len(tasks) else None
    for index, task in enumerate(tasks, start=1):
        evidence_id = str(task["evidence_id"])
        if evidence_id in completed:
            continue
        if generator is None:
            raise RuntimeError("Local LLM was not initialized for an incomplete extraction run.")
        generation = generator.generate_json(
            system_prompt=SYSTEM_PROMPT,
            payload={
                "evidence_id": evidence_id,
                "source_text": task["source_text"],
                "target_node_id": task["target_node_id"],
                "target_name": task["target_name"],
                "target_role": task["target_role"],
            },
            max_new_tokens=768,
        )
        decision = validate_generation(task, generation)
        _append_jsonl(state_path, decision)
        completed[evidence_id] = decision
        print(json.dumps({"processed": index, "total": len(tasks)}, sort_keys=True), flush=True)

    ordered = [completed[str(task["evidence_id"])] for task in tasks]
    validated_entities = [
        entity for row in ordered for entity in row.get("validated_entities") or []
    ]
    unresolved_for_resolution = [
        {
            "mention_id": entity["mention_id"],
            "evidence_id": entity["evidence_id"],
            "entity_type": "Compound",
            "mention_text": entity["mention_text"],
            "normalized_text": normalize_text(entity["mention_text"]),
            "candidate_identifiers": [],
            "rejection_reason": "requires_public_identifier_resolution",
            "recognition_confidence": entity["confidence"],
        }
        for entity in validated_entities
        if entity["entity_type"] == "specific_compound"
    ]
    contextual_entities = [
        entity for entity in validated_entities if entity["entity_type"] != "specific_compound"
    ]
    _atomic_jsonl(output_dir / "complex_entity_decisions.jsonl", ordered)
    _atomic_jsonl(output_dir / "compound_mentions_for_resolution.jsonl", unresolved_for_resolution)
    _atomic_jsonl(output_dir / "noncompound_context_entities.jsonl", contextual_entities)
    type_counts = Counter(entity["entity_type"] for entity in validated_entities)
    summary = {
        "accepted": len(ordered) == len(tasks),
        "tasks_path": str(tasks_path),
        "tasks_sha256": _sha256(tasks_path),
        "model_name": model_name,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "task_count": len(tasks),
        "validated_entity_count": len(validated_entities),
        "entity_type_counts": dict(sorted(type_counts.items())),
        "compound_resolution_candidate_count": len(unresolved_for_resolution),
        "noncompound_context_entity_count": len(contextual_entities),
        "invalid_entity_count": sum(len(row.get("rejected_entities") or []) for row in ordered),
        "invalid_json_count": sum(bool(row.get("generation_error")) for row in ordered),
    }
    _atomic_json(output_dir / "complex_entity_summary.json", summary)
    return summary


def validate_generation(
    task: dict[str, Any],
    generation: StructuredGeneration,
) -> dict[str, Any]:
    base = {
        "evidence_id": task["evidence_id"],
        "source_text_hash": task["source_text_hash"],
        "raw_model_output": generation.raw_text,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "validated_entities": [],
        "rejected_entities": [],
        "generation_error": generation.error,
    }
    if generation.parsed is None:
        return base
    entities = generation.parsed.get("entities")
    if not isinstance(entities, list):
        return {**base, "generation_error": "entities_must_be_a_list"}
    seen: set[tuple[str, str]] = set()
    for item in entities:
        normalized_item, adjustment = apply_deterministic_type_constraints(item)
        errors = validate_entity(task, normalized_item)
        if isinstance(normalized_item, dict):
            key = (
                normalize_text(normalized_item.get("mention_text")),
                str(normalized_item.get("entity_type") or ""),
            )
            if key in seen:
                errors.append("duplicate_entity")
            seen.add(key)
        if errors:
            base["rejected_entities"].append(
                {
                    "entity": normalized_item,
                    "type_adjustment": adjustment,
                    "validation_errors": sorted(set(errors)),
                }
            )
            continue
        mention_text = str(normalized_item["mention_text"])
        start = str(task["source_text"]).find(mention_text)
        base["validated_entities"].append(
            {
                "mention_id": _stable_mention_id(
                    str(task["evidence_id"]), mention_text, str(normalized_item["entity_type"])
                ),
                "evidence_id": str(task["evidence_id"]),
                "mention_text": mention_text,
                "start_char": start,
                "end_char": start + len(mention_text),
                "entity_type": normalized_item["entity_type"],
                "role": normalized_item["role"],
                "supporting_quote": normalized_item["supporting_quote"],
                "confidence": float(normalized_item["confidence"]),
                "reason": normalized_item["reason"],
                "recognition_method": "qwen_complex_ddi_entity_extraction",
                "type_adjustment": adjustment,
            }
        )
    return base


def apply_deterministic_type_constraints(item: Any) -> tuple[Any, str | None]:
    if not isinstance(item, dict) or item.get("entity_type") != "specific_compound":
        return item, None
    mention = str(item.get("mention_text") or "").strip()
    if re.fullmatch(r"[A-Z]{2,5}", mention):
        return {**item, "entity_type": "unknown_abbreviation"}, (
            "short_unexpanded_uppercase_token_reclassified"
        )
    if DRUG_CLASS_TERMS.search(mention):
        return {**item, "entity_type": "drug_class"}, "drug_class_term_reclassified"
    return item, None


def validate_entity(task: dict[str, Any], item: Any) -> list[str]:
    if not isinstance(item, dict):
        return ["entity_must_be_an_object"]
    errors = []
    text = str(task["source_text"])
    mention = str(item.get("mention_text") or "")
    quote = str(item.get("supporting_quote") or "")
    if not mention or mention not in text:
        errors.append("mention_not_verbatim")
    if not quote or quote not in text:
        errors.append("supporting_quote_not_verbatim")
    elif mention and mention not in quote:
        errors.append("mention_missing_from_quote")
    if normalize_text(mention) == normalize_text(task.get("target_name")):
        errors.append("target_compound_cannot_be_counterpart_candidate")
    if item.get("entity_type") not in ENTITY_TYPES:
        errors.append("invalid_entity_type")
    if item.get("role") not in {
        "victim_drug",
        "perpetrator_drug",
        "interacting_product",
        "unknown",
    }:
        errors.append("invalid_entity_role")
    try:
        confidence = float(item.get("confidence"))
    except (TypeError, ValueError):
        confidence = -1.0
    if confidence < MIN_CONFIDENCE or confidence > 1.0:
        errors.append("confidence_below_threshold_or_invalid")
    if not str(item.get("reason") or "").strip():
        errors.append("reason_missing")
    return errors


def _stable_mention_id(evidence_id: str, mention: str, entity_type: str) -> str:
    digest = hashlib.sha256(f"{evidence_id}\x1f{mention}\x1f{entity_type}".encode()).hexdigest()
    return f"complex-mention:{digest[:24]}"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            yield value


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
