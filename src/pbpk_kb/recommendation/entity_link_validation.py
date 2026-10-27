from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Protocol

from pbpk_kb.recommendation.entity_extraction import normalize_text
from pbpk_kb.recommendation.kg_models import utc_now
from pbpk_kb.recommendation.local_llm import DEFAULT_LOCAL_LLM, LocalStructuredGenerator, StructuredGeneration


PROMPT_VERSION = "pbpk-entity-context-reranker-1.0.0"
VALIDATOR_VERSION = "pbpk-entity-link-validator-1.2.0"
SYSTEM_PROMPT = """You validate biomedical entity links for a PBPK knowledge graph.
Use only the supplied evidence contexts and candidate list.
Return one JSON object with exactly these keys:
decision, selected_candidate_id, supporting_quote, entity_role, confidence, reason.
decision must be accepted, ambiguous, or unresolved.
For accepted, selected_candidate_id must exactly equal one supplied candidate_id.
supporting_quote must be one exact contiguous quote from a supplied context and contain the mention.
Reject drug classes, foods, herbs, combination regimens, malformed tokens, and abbreviations when the supplied
context does not uniquely establish that they denote the selected specific chemical or drug concept.
Do not create a name, identifier, candidate, fact, or relation. Do not use outside knowledge."""


class StructuredGenerator(Protocol):
    model_name: str

    def generate_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        max_new_tokens: int = 768,
    ) -> StructuredGeneration: ...


def main() -> None:
    parser = argparse.ArgumentParser(description="Context-rerank PubChem/RxNorm entity candidates.")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--resolver-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_LOCAL_LLM)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    generator = LocalStructuredGenerator(args.model)
    summary = validate_entity_links(
        evidence_path=args.evidence,
        resolver_decisions_path=args.resolver_decisions,
        output_dir=args.output_dir,
        generator=generator,
        min_confidence=args.min_confidence,
        reset=args.reset,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def validate_entity_links(
    *,
    evidence_path: Path,
    resolver_decisions_path: Path,
    output_dir: Path,
    generator: StructuredGenerator,
    min_confidence: float,
    reset: bool,
) -> dict[str, Any]:
    if reset and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {str(row["evidence_id"]): row for row in _iter_jsonl(evidence_path)}
    resolver_decisions = list(_iter_jsonl(resolver_decisions_path))
    decisions: list[dict[str, Any]] = []
    catalog: dict[str, dict[str, Any]] = {}
    model_call_count = 0

    for index, resolver_decision in enumerate(resolver_decisions, start=1):
        contexts = _contexts_for(resolver_decision, evidence)
        candidates = _candidate_options(resolver_decision)
        hard_rejection = _hard_rejection_reason(str(resolver_decision.get("name") or ""))
        automatic = _automatic_dual_source_decision(resolver_decision, contexts, candidates)
        if hard_rejection is not None:
            decision = {
                "decision": "unresolved",
                "selected_candidate_id": None,
                "supporting_quote": _first_verbatim_context(str(resolver_decision.get("name") or ""), contexts),
                "entity_role": _single_role(resolver_decision),
                "confidence": 0.0,
                "reason": hard_rejection,
                "validation_errors": [],
                "decision_method": "deterministic_noncompound_or_underspecified_rejection",
                "raw_model_output": None,
            }
        elif automatic is not None:
            decision = automatic
        elif not candidates:
            decision = {
                "decision": "unresolved",
                "selected_candidate_id": None,
                "supporting_quote": None,
                "entity_role": _single_role(resolver_decision),
                "confidence": 0.0,
                "reason": "no_identifier_candidate_available_for_context_reranking",
                "validation_errors": [],
                "decision_method": "deterministic_no_candidate",
                "raw_model_output": None,
            }
        else:
            model_call_count += 1
            result = generator.generate_json(
                system_prompt=SYSTEM_PROMPT,
                payload={
                    "mention": resolver_decision["name"],
                    "roles": resolver_decision.get("roles") or [],
                    "contexts": contexts,
                    "candidates": candidates,
                },
            )
            decision = _validate_model_decision(
                result,
                mention=str(resolver_decision["name"]),
                contexts=contexts,
                candidates=candidates,
                min_confidence=min_confidence,
            )
        record = {
            "resolver_candidate": resolver_decision,
            "context_rerank": decision,
            "prompt_version": PROMPT_VERSION,
            "validator_version": VALIDATOR_VERSION,
            "model_name": generator.model_name,
        }
        decisions.append(record)
        if decision["decision"] == "accepted":
            selected = next(
                item for item in candidates if item["candidate_id"] == decision["selected_candidate_id"]
            )
            entry = selected["catalog_entry"]
            catalog[entry["node_id"]] = entry
        if index % 10 == 0 or index == len(resolver_decisions):
            print(
                json.dumps(
                    {
                        "validated_candidates": index,
                        "total": len(resolver_decisions),
                        "model_calls": model_call_count,
                    }
                ),
                flush=True,
            )

    status_counts = Counter(item["context_rerank"]["decision"] for item in decisions)
    method_counts = Counter(item["context_rerank"]["decision_method"] for item in decisions)
    summary = {
        "accepted": len(decisions) == len(resolver_decisions),
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "model_name": generator.model_name,
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
        "resolver_decisions_path": str(resolver_decisions_path),
        "resolver_decisions_sha256": _sha256(resolver_decisions_path),
        "input_candidate_count": len(resolver_decisions),
        "accepted_catalog_count": len(catalog),
        "model_call_count": model_call_count,
        "status_counts": dict(sorted(status_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "generated_at": utc_now(),
    }
    _atomic_jsonl(output_dir / "validated_external_entity_catalog.jsonl", sorted(catalog.values(), key=lambda x: x["node_id"]))
    _atomic_jsonl(output_dir / "entity_link_validation_decisions.jsonl", decisions)
    _atomic_json(output_dir / "entity_link_validation_summary.json", summary)
    return summary


def _candidate_options(resolver_decision: dict[str, Any]) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    entry = resolver_decision.get("catalog_entry")
    if isinstance(entry, dict):
        candidate = _option_from_entry(entry, resolver_decision)
        output[candidate["candidate_id"]] = candidate
    pubchem = resolver_decision.get("pubchem") if isinstance(resolver_decision.get("pubchem"), dict) else {}
    for match in pubchem.get("matches") or []:
        cid = str(match.get("cid") or "").strip()
        if not cid:
            continue
        entry = {
            "node_id": f"Compound:PUBLIC:{cid}",
            "node_type": "Compound",
            "preferred_name": str(match.get("title") or resolver_decision["name"]),
            "aliases": sorted(
                set([str(resolver_decision["name"]), *(str(item) for item in match.get("matching_synonyms") or [])]),
                key=str.lower,
            ),
            "standard_identifiers": {
                key: value
                for key, value in {
                    "PUBCHEM.COMPOUND": cid,
                    "INCHIKEY": str(match.get("inchi_key") or ""),
                }.items()
                if value
            },
            "resolution_method": "pubchem_exact_context_reranked",
            "resolution_query": str(resolver_decision["name"]),
        }
        candidate = {
            "candidate_id": entry["node_id"],
            "preferred_name": entry["preferred_name"],
            "standard_identifiers": entry["standard_identifiers"],
            "resolver_sources": ["PubChem"],
            "catalog_entry": entry,
        }
        output[candidate["candidate_id"]] = candidate
    return sorted(output.values(), key=lambda item: item["candidate_id"])


def _option_from_entry(entry: dict[str, Any], resolver_decision: dict[str, Any]) -> dict[str, Any]:
    sources = []
    if resolver_decision.get("pubchem", {}).get("status") == "resolved":
        sources.append("PubChem")
    if resolver_decision.get("rxnorm", {}).get("status") == "resolved":
        sources.append("RxNorm")
    return {
        "candidate_id": str(entry["node_id"]),
        "preferred_name": str(entry["preferred_name"]),
        "standard_identifiers": dict(entry["standard_identifiers"]),
        "resolver_sources": sources,
        "catalog_entry": entry,
    }


def _contexts_for(
    resolver_decision: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    output = []
    seen = set()
    for evidence_id in resolver_decision.get("evidence_ids") or []:
        row = evidence.get(str(evidence_id))
        text = str(row.get("source_text") or "") if row else ""
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(
            {
                "evidence_id": str(evidence_id),
                "source_id": str(row.get("source_id") or ""),
                "source_location": str(row.get("source_location") or ""),
                "text": text,
            }
        )
    return output[:5]


def _automatic_dual_source_decision(
    resolver_decision: dict[str, Any],
    contexts: list[dict[str, str]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    pubchem_ok = resolver_decision.get("pubchem", {}).get("status") == "resolved"
    rxnorm_ok = resolver_decision.get("rxnorm", {}).get("status") == "resolved"
    mention = str(resolver_decision.get("name") or "")
    quote = _first_verbatim_context(mention, contexts)
    if not pubchem_ok or not rxnorm_ok or len(candidates) != 1 or quote is None or _lexically_risky(mention):
        return None
    return {
        "decision": "accepted",
        "selected_candidate_id": candidates[0]["candidate_id"],
        "supporting_quote": quote,
        "entity_role": _single_role(resolver_decision),
        "confidence": 1.0,
        "reason": "pubchem_rxnorm_unique_exact_agreement_with_verbatim_context",
        "validation_errors": [],
        "decision_method": "deterministic_dual_source_agreement",
        "raw_model_output": None,
    }


def _validate_model_decision(
    result: StructuredGeneration,
    *,
    mention: str,
    contexts: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    min_confidence: float,
) -> dict[str, Any]:
    if result.parsed is None:
        return _failed_model_decision(result, ["model_output_not_valid_json"])
    value = result.parsed
    decision = str(value.get("decision") or "").strip().lower()
    selected = value.get("selected_candidate_id")
    quote = str(value.get("supporting_quote") or "")
    confidence = _as_confidence(value.get("confidence"))
    candidate_ids = {item["candidate_id"] for item in candidates}
    context_texts = [item["text"] for item in contexts]
    errors = []
    if decision not in {"accepted", "ambiguous", "unresolved"}:
        errors.append("invalid_decision")
    if decision == "accepted":
        if selected not in candidate_ids:
            errors.append("selected_candidate_not_in_input")
        else:
            selected_candidate = next(item for item in candidates if item["candidate_id"] == selected)
            if not _candidate_matches_mention(selected_candidate, mention):
                errors.append("selected_candidate_name_mismatch")
        if not quote or not any(quote in text for text in context_texts):
            errors.append("supporting_quote_not_verbatim")
        if normalize_text(mention) not in normalize_text(quote):
            errors.append("mention_missing_from_quote")
        if confidence < min_confidence:
            errors.append("confidence_below_threshold")
    if errors:
        return _failed_model_decision(result, errors)
    return {
        "decision": decision,
        "selected_candidate_id": selected if decision == "accepted" else None,
        "supporting_quote": quote or None,
        "entity_role": str(value.get("entity_role") or "") or None,
        "confidence": confidence,
        "reason": str(value.get("reason") or ""),
        "validation_errors": [],
        "decision_method": "qwen_context_reranker",
        "raw_model_output": result.raw_text,
    }


def _failed_model_decision(result: StructuredGeneration, errors: list[str]) -> dict[str, Any]:
    return {
        "decision": "unresolved",
        "selected_candidate_id": None,
        "supporting_quote": None,
        "entity_role": None,
        "confidence": 0.0,
        "reason": result.error or "deterministic_model_output_validation_failed",
        "validation_errors": errors,
        "decision_method": "qwen_context_reranker_rejected",
        "raw_model_output": result.raw_text,
    }


def _lexically_risky(value: str) -> bool:
    return _hard_rejection_reason(value) is not None


def _hard_rejection_reason(value: str) -> str | None:
    normalized = normalize_text(value)
    tokens = set(normalized.split())
    if re.fullmatch(r"[A-Z]{2,6}", value.strip()) and not re.search(r"\d", value):
        return "unexpanded_short_abbreviation_requires_unique_context_resolution"
    if tokens & {
        "antibiotic",
        "antibiotics",
        "antimycotic",
        "antimycotics",
        "blocker",
        "blockers",
        "class",
        "components",
        "quinolone",
        "statin",
    }:
        return "drug_class_term_cannot_enter_compound_nodes"
    if tokens & {"extract", "food", "herb", "herbal"}:
        return "botanical_or_food_product_requires_noncompound_ontology_resolution"
    return None


def _candidate_matches_mention(candidate: dict[str, Any], mention: str) -> bool:
    entry = candidate.get("catalog_entry") if isinstance(candidate.get("catalog_entry"), dict) else {}
    names = {
        str(candidate.get("preferred_name") or ""),
        str(entry.get("preferred_name") or ""),
        *(str(item) for item in entry.get("aliases") or []),
    }
    wanted = normalize_text(mention)
    return bool(wanted) and any(normalize_text(value) == wanted for value in names)


def _first_verbatim_context(mention: str, contexts: list[dict[str, str]]) -> str | None:
    pattern = re.compile(re.escape(mention), flags=re.IGNORECASE)
    for context in contexts:
        text = context["text"]
        if pattern.search(text):
            return text
    return None


def _single_role(value: dict[str, Any]) -> str | None:
    roles = [str(item) for item in value.get("roles") or [] if item]
    return roles[0] if len(set(roles)) == 1 else None


def _as_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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
